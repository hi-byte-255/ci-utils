#!/usr/bin/env python3
"""authenticity_pass.py — Wave 13 tier-2: frontier probing + spoof/decoy screening on the GOLD set.

Runs ON THE RUNNERS. Never locally (operator egress guard).
Writes validated_auth_<shard>.json artifacts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

import aiohttp

# ---------------------------------------------------------------- egress guard
if not os.environ.get("GITHUB_ACTIONS") and not os.environ.get("WAVE13_LOCAL_SPOTCHECK"):
    sys.exit("FATAL: bulk auth pass must run on GitHub Actions runners (set WAVE13_LOCAL_SPOTCHECK only for <=5 hosts).")

HONEYPOT_IDS = {"chatcmpl-12c423"}
HONEYPOT_CREATED = {1677858242}
DECOY_MARKERS = ("synthetic diagnostic", "no model was executed")

# Frontier families we care about, with the canonical probe target per family
FRONTIER = {
    "claude": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-opus-4-5", "claude-sonnet-4-5", "claude-3-5-sonnet-20241022"],
    "gpt": ["gpt-5", "gpt-5.1", "gpt-4o", "gpt-4.1", "o3", "o4-mini"],
    "gemini": ["gemini-3-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash"],
    "deepseek": ["deepseek-v4", "deepseek-r1", "deepseek-v3.2"],
    "qwen": ["qwen3-235b-a22b", "qwen3-coder-480b", "qwen3-max"],
    "glm": ["glm-5", "glm-4.6", "glm-4.5"],
    "kimi": ["kimi-k3", "kimi-k2"],
    "llama": ["llama-4-scout", "llama-4-maverick", "llama-3.3-70b"],
}

SMALL_LOCAL = ("llama3", "qwen2", "smollm", "tinyllama", "phi-3", "gemma-2b", "gemma:2b", "qwen1.5", "stablelm")

ARITH_PROMPT = "Reply with ONLY the number: 23*17"
ARITH_EXPECT = "391"
NONCE_PROMPT = ("Reply with ONLY this token, exactly: ZX7QK4. "
                "Then on a new line give one unusual fact about tardigrades.")


def strip_think(text: str) -> str:
    return re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text or "", flags=re.S | re.I).strip()


def check_stub(text: str, raw: dict) -> str | None:
    low = (text or "").lower()
    for m in DECOY_MARKERS:
        if m in low:
            return "SYNTHETIC_DECOY"
    if raw.get("id") in HONEYPOT_IDS or raw.get("created") in HONEYPOT_CREATED:
        return "HONEYPOT_EPOCH"
    if len(text.strip()) < 2 and not raw.get("choices", [{}])[0].get("message", {}).get("tool_calls"):
        return "stub_or_empty"
    return None


async def post(session, url, payload, timeout):
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
        body = await r.text()
        try:
            js = json.loads(body)
        except Exception:
            js = {"_raw": body[:600]}
        return r.status, js


def extract(js: dict) -> tuple[str, bool]:
    ch = (js.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
    tool_calls = bool(msg.get("tool_calls"))
    return strip_think(content if isinstance(content, str) else json.dumps(content)), tool_calls


async def probe_model(session, base, model, timeout):
    """One frontier-authenticity probe: 2 prompts + response envelope forensics."""
    url = base.rstrip("/") + "/chat/completions"
    out = {"model": model, "verified": False, "reasons": [], "echo": None}

    st, js = await post(session, url, {"model": model, "messages": [{"role": "user", "content": ARITH_PROMPT}],
                                       "max_tokens": 1024}, timeout)
    if st >= 400:
        out["reasons"].append(f"http_{st}")
        return out
    text, tools = extract(js)
    out["echo"] = js.get("model")
    stub = check_stub(text, js)
    if stub:
        out["reasons"].append(stub)
        return out

    if not re.search(rf"(?<!\d){ARITH_EXPECT}(?!\d)", text):
        out["reasons"].append("arithmetic_fail")
    if tools:
        out["reasons"].append("tool_calls_only")
        out["tool_call_host"] = True

    st2, js2 = await post(session, url, {"model": model, "messages": [{"role": "user", "content": NONCE_PROMPT}],
                                         "max_tokens": 1024}, timeout)
    if st2 < 400:
        t2, _ = extract(js2)
        if "ZX7QK4" not in t2:
            out["reasons"].append("nonce_echo_fail")
        if len(t2) < 40:
            out["reasons"].append("nonce_terse")

    # model-echo forensics: frontier label answered by a small local backend
    echo = (out.get("echo") or "").lower()
    if echo and any(s in echo for s in SMALL_LOCAL):
        out["reasons"].append(f"FAKE_MODEL_SPOOF(echo={echo})")

    out["verified"] = not [r for r in out["reasons"]
                           if r.startswith(("arithmetic_fail", "FAKE_MODEL_SPOOF", "SYNTHETIC", "HONEYPOT", "nonce_echo_fail"))]
    return out


async def handle_host(session, row, sem, timeout):
    host = row.get("host")
    ip, _, port = host.partition(":")
    scheme = "https" if row.get("tier_port_hint") == "tls" else "http"
    base = f"{scheme}://{host}"
    path = row.get("endpoint_path") or "/v1/models"
    apibase = base + path.rsplit("/models", 1)[0] if path.endswith("/models") else base + "/v1"

    async with sem:
        rec = {"host": host, "ip": ip, "port": port, "api_base": apibase,
               "authenticity": None, "frontier": [], "decoy": None, "errors": []}

        # ---- 1. decoy / honeypot screen on the advertised model list
        try:
            async with session.get(apibase + "/models", timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                body = await r.text()
                js = json.loads(body) if body.strip().startswith("{") else {}
        except Exception as e:  # noqa: BLE001
            js = {}
            rec["errors"].append(f"models:{type(e).__name__}")

        advertised = [m.get("id") for m in (js.get("data") or []) if m.get("id")]
        low = json.dumps(js).lower()
        if any(d in low for d in DECOY_MARKERS):
            rec["decoy"] = "SYNTHETIC_DECOY"

        # ---- 2. frontier probing: family-matched then generic
        targets = []
        for fam, probes in FRONTIER.items():
            if any(fam in a.lower() for a in advertised):
                targets.extend([m for m in advertised
                                if fam in m.lower() and not any(j in m.lower() for j in ("proxy", "openrouter", "lora", "finetune"))][:2])
        if not targets:
            for probes in FRONTIER.values():
                targets.extend([m for m in advertised if m in probes][:1])
        targets = list(dict.fromkeys(targets))[:6]

        if not targets and advertised:
            targets = advertised[:3]

        for t in targets:
            try:
                rec["frontier"].append(await probe_model(session, apibase, t, timeout))
            except Exception as e:  # noqa: BLE001
                rec["errors"].append(f"{t}:{type(e).__name__}")

        verified = [f["model"] for f in rec["frontier"] if f.get("verified")]
        rec["verified_models"] = verified
        rec["authenticity"] = ("VERIFIED_FRONTIER" if verified
                               else "DECOY" if rec["decoy"]
                               else "UNVERIFIED")
        return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hosts", required=True)
    ap.add_argument("-o", "--output", default="validated_auth.json")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.hosts) if l.strip()]
    rows = [r for r in rows if r.get("host")]
    if not rows:
        sys.exit("no hosts")

    sem = asyncio.Semaphore(args.workers)
    conn = aiohttp.TCPConnector(limit=args.workers * 2, ssl=False)
    async with aiohttp.ClientSession(connector=conn) as s:
        results = await asyncio.gather(*[handle_host(s, r, sem, args.timeout) for r in rows],
                                       return_exceptions=True)

    out = []
    for r in results:
        if isinstance(r, Exception):
            out.append({"error": str(r)})
        else:
            out.append(r)
    json.dump(out, open(args.output, "w"), indent=1)

    from collections import Counter
    c = Counter(x.get("authenticity") for x in out)
    v = sum(len(x.get("verified_models") or []) for x in out)
    print(f"hosts={len(out)} authenticity={dict(c)} verified_models={v}")


if __name__ == "__main__":
    asyncio.run(main())
