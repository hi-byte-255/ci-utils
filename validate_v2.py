#!/usr/bin/env python3
"""
validate_v2.py — Turbo Hardened Multi-Model Router Validator (Zero-Ollama, Async Swarm).

Optimizations:
  1. Parallel Multi-Path Discovery: Probes /v1/models, /api/v1/models, /openai/v1/models, /model/info
     in parallel via asyncio.gather with a 3.5s timeout (drops dead host latency from 48s to 3.5s).
  2. Non-Chat Model Filtering: Skips embedding, rerank, tts, whisper, dall-e, clip, flux, cogview, sora, midjourney.
  3. Fast Route Snap (Early Exit): If prompt 1 returns 401, 403, 404, 502, or 503, immediately snaps
     that route without running the remaining 7 test calls.
  4. Consecutive-Timeout Abort: If prompts 1 AND 2 both time out, skips remaining checks entirely.
  5. Streaming Hardening: TTFT ceiling (4s), chunk counts only on real content, [DONE] break,
     partial results preserved on timeout, early exit at 5 chunks.
  6. Host-Level Total Budget: 90s cap per host — partial results saved on budget exhaustion.
  7. Incremental JSONL Ledger: Each completed host appended to .jsonl immediately.
  8. Arithmetic Boundary Check: Word-boundary regex on 391 (no false positives from timestamps/IDs).
  9. Real-Time Streaming Logs: Prints each host result to stdout immediately upon completion.
 10. High Concurrency: Defaults to 32 concurrent workers.
 11. Strict Zero-Ollama Filter: Hard-drops port 11434 (unless --include-ollama).

Usage:
    python3 validate_v2.py --hosts "ip:port,ip:port" [--models 20] [--workers 32] [--timeout 10] [-o out.json]
"""
import argparse, asyncio, json, time, sys, re, os
import aiohttp

SOFT_PROMPTS = [
    ("fizzbuzz", "Write a short Python function fizzbuzz(n) that returns 'Fizz', 'Buzz', 'FizzBuzz', or str(n). Output code only."),
    ("logic_riddle", "If I have 10 sheep and all but 7 run away, how many sheep do I have left? Respond with just the single number."),
    ("code_palindrome", "Write a Python function is_palindrome(s) returning boolean. Output code only."),
    ("creative_short", "Write a 1-sentence thought about the ocean at night."),
]
VARIANCE_PROMPT = "Write a 1-sentence thought about the ocean at night."
ARITH_PROMPT = "What is 17*23? Respond with just the number."
ARITH_RE = re.compile(r"(?<!\d)391(?!\d)")
STREAM_PROMPT = "Count from 1 to 30, one number per line."
TOPIC_PROMPT = "Write a 3-line haiku about a teal bicycle. It must mention a bicycle or bike."
TOPIC_KEYWORDS = ["bicycle", "bike", "pedal", "wheel", "teal", "ride", "cycle", "spoke", "chain"]
TTFT_CEILING = 4.0  # seconds — max time to first real content chunk
STREAM_EARLY_CHUNKS = 5  # exit streaming once this many content chunks received
HOST_BUDGET = 90.0  # seconds — max total time per host
CONNECT_TIMEOUT = 2.0  # seconds — TCP connect timeout per request

CANDIDATE_PATHS = [
    ("/v1/models",        "/v1/chat/completions"),
    ("/api/v1/models",    "/api/v1/chat/completions"),
    ("/api/models",       "/api/v1/chat/completions"),
    ("/openai/v1/models", "/openai/v1/chat/completions"),
    ("/v1beta/models",    "/v1/chat/completions"),
    ("/model/info",       "/v1/chat/completions"),
]

DEFAULT_TEMPLATE_KEYS = (
    "Bearer sk-1234",
    "Bearer sk-123456",
    "Bearer EMPTY",
)

FRONTIER_PRIORITY_PREFIXES = (
    "[按量", "按量ccmax", "小e-", "单间-", "claude-opus", "claude-sonnet-4",
    "anthropic.claude", "us.openai.gpt-6", "gpt-6", "gpt-5", "o3", "o4",
    "gemini-2.5-pro", "gemini-3", "deepseek-v4", "deepseek-r1", "kimi-k2",
    "kimi-k3", "qwen3-235b", "qwen3-coder", "qwen3.8", "glm-5",
)

SMALL_LOCAL_SPOOF_FAMILIES = (
    "llama3:8b", "llama-3-8b", "llama3.2:1b", "llama3.2:3b", "qwen2:0.5b",
    "qwen2:1.5b", "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "smollm",
    "tinyllama", "phi-3-mini", "gemma:2b", "gemma2:2b", "mistral:7b",
)


def sort_models_by_priority(models: list[str]) -> list[str]:
    """Place Tea-channel prefixes and frontier models first so host budget is spent on high-value models."""
    def _rank(m: str) -> tuple[int, str]:
        low = m.lower()
        for idx, pref in enumerate(FRONTIER_PRIORITY_PREFIXES):
            if pref in low:
                return (idx, m)
        return (len(FRONTIER_PRIORITY_PREFIXES) + 1, m)
    return sorted(models, key=_rank)

NON_CHAT_KEYWORDS = [
    "embedding", "embed", "rerank", "bge-", "tts", "whisper",
    "dall-e", "stable-diffusion", "clip", "moderation", "vl-embedding",
    "flux", "cogview", "sora", "midjourney"
]

STUB_MARKERS = [
    "as an ai language model", "i'm a helpful ai assistant", "i'm just an ai",
    "regarding", "i'd be happy to help", "let me think about this",
    "i'll do my best to assist", "qyora-gate", "久辰", "owner of the router",
    "i cannot fulfill this request", "token plan not support",
    "grey water folds the morning in", "the sea breathes slow", "the ocean breathes slow",
    "the tide turns slowly", "water folds the morning in",
    "hello! i'm doing well, thank you", "ironclad-7b", "nexus-7b-sec", "gpus4all",
    "no active credentials for provider", "我会一步步为你拆解", "响应。\n\n我会一步步",
]

THINK_STRIP_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>\s*", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    if not text:
        return ""
    cleaned = THINK_STRIP_RE.sub("", text).strip()
    return cleaned if cleaned else text.strip()

# 2026-10-09: personas injected by gateways that run their OWN agent/bot prompt.
# A reply that names one of these is the host's injected identity, not the model's
# self-ID — flag it so authenticity classification isn't fooled.
INJECTED_PERSONA_MARKERS = (
    "kiro", "aembit", "bizalign", "水聖人", "water sage", "soul vault", "best3",
)


def check_stub(text: str) -> bool:
    if not text:
        return True
    low = text.lower()
    return any(m in low for m in STUB_MARKERS)

def is_chat_model(m_name: str) -> bool:
    low = m_name.lower()
    return not any(k in low for k in NON_CHAT_KEYWORDS)


async def chat(session, base, chat_path, model, prompt, timeout, stream=False, max_tokens=2048, headers=None, extra_payload=None):
    """One chat call. Returns dict(status, content, ttft, chunks, elapsed) or error info."""
    t0 = time.time()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream
    }
    if extra_payload:
        payload.update(extra_payload)
    url = f"{base}{chat_path}"
    # Initialize streaming vars outside try so outer except can check them
    ttft, chunks, text = None, 0, []
    try:
        async with asyncio.timeout(timeout):
            if not stream:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT)) as r:
                    if r.status != 200:
                        err_txt = ""
                        if r.status in (400, 502, 503):
                            try:
                                err_txt = (await r.text())[:240]
                            except Exception:
                                pass
                        return {"status": r.status, "error_body": err_txt}
                    try:
                        body = await r.json(content_type=None)
                    except Exception:
                        return {"status": "bad_json"}
                    if isinstance(body, dict) and (body.get("id") == "chatcmpl-12c423" or body.get("created") == 1677858242):
                        return {"status": 200, "content": "hello! i'm doing well, thank you", "alibaba_honeypot": True, "elapsed": time.time() - t0}
                    choices = body.get("choices") if isinstance(body, dict) else []
                    if not choices or not isinstance(choices, list):
                        return {"status": "no_choices"}
                    ch0 = choices[0] if isinstance(choices[0], dict) else {}
                    msg = ch0.get("message", {}) if isinstance(ch0.get("message"), dict) else {}
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or ""
                    full_text = (strip_thinking(content) or reasoning).strip()
                    # 2026-10-09 lesson: a gateway that injects its OWN toolset answers
                    # with tool_calls + EMPTY content. Scoring that as a stub is a FALSE
                    # NEGATIVE — the host is alive and real, it just wants a tool run.
                    # Capture the call names instead of discarding them.
                    tcs = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
                    tc_names = [c.get("function", {}).get("name") for c in tcs
                                if isinstance(c, dict) and isinstance(c.get("function"), dict)]
                    tc_names = [n for n in tc_names if n]
                    # Hidden system prompt: we send NO system message, so any prompt_tokens
                    # beyond our own payload means the gateway injected one. Observed up to
                    # 6,708 tokens this way during the 2026-10-09 payload triage.
                    _usage = body.get("usage")
                    usage = _usage if isinstance(_usage, dict) else {}
                    # Owner-ID leak: some gateways echo upstream provider headers.
                    _ef = body.get("extra_fields")
                    ef = _ef if isinstance(_ef, dict) else {}
                    return {"status": 200, "content": full_text, "raw_msg": msg,
                            "raw_content": content,
                            "resp_model": body.get("model") if isinstance(body, dict) else None,
                            "tool_calls": tc_names,
                            "finish_reason": ch0.get("finish_reason"),
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "header_leak": "provider_response_headers" in ef,
                            "elapsed": time.time() - t0}

            # Streaming mode
            ttft, chunks, text = None, 0, []
            done = False
            try:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT)) as r:
                    if r.status != 200:
                        return {"status": r.status}
                    async for line in r.content:
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        if line == b"data: [DONE]":
                            done = True
                            break
                        try:
                            delta = json.loads(line[5:])["choices"][0].get("delta", {})
                            d_text = delta.get("content") or delta.get("reasoning_content") or ""
                        except Exception:
                            continue
                        if not d_text:
                            continue
                        # Only count chunks and set ttft on real content
                        if ttft is None:
                            ttft = time.time() - t0
                            if ttft > TTFT_CEILING:
                                break  # TTFT exceeded — snap
                        chunks += 1
                        text.append(d_text)
                        if chunks >= STREAM_EARLY_CHUNKS:
                            done = True
                            break
            except (asyncio.TimeoutError, TimeoutError):
                # Partial results on timeout — preserve what we got
                elapsed = time.time() - t0
                if chunks > 0:
                    return {
                        "status": 200,
                        "content": "".join(text).strip(),
                        "ttft": ttft,
                        "chunks": chunks,
                        "elapsed": elapsed,
                        "partial": True
                    }
                return {"status": "timeout"}
            except Exception:
                # Stream-level error — preserve partial results
                elapsed = time.time() - t0
                if chunks > 0:
                    return {
                        "status": 200,
                        "content": "".join(text).strip(),
                        "ttft": ttft,
                        "chunks": chunks,
                        "elapsed": elapsed,
                        "partial": True
                    }
                return {"status": "error"}
            elapsed = time.time() - t0
            return {
                "status": 200,
                "content": "".join(text).strip(),
                "ttft": ttft,
                "chunks": chunks,
                "elapsed": elapsed
            }
    except asyncio.TimeoutError:
        # Outer timeout fired — check if streaming collected partial results
        if stream and chunks > 0:
            return {
                "status": 200,
                "content": "".join(text).strip(),
                "ttft": ttft,
                "chunks": chunks,
                "elapsed": time.time() - t0,
                "partial": True
            }
        return {"status": "timeout"}
    except Exception as e:
        # Outer exception — preserve partial streaming results
        if stream and chunks > 0:
            return {
                "status": 200,
                "content": "".join(text).strip(),
                "ttft": ttft,
                "chunks": chunks,
                "elapsed": time.time() - t0,
                "partial": True
            }
        return {"status": "error", "error": str(e)[:100]}


def _is_frontier_spoof(req_model: str, resp_model: str | None) -> bool:
    if not resp_model:
        return False
    req_low = req_model.lower()
    resp_low = str(resp_model).lower()
    if any(f in req_low for f in ("claude", "gpt-5", "gpt-6", "o3", "o4", "gemini-2.5", "gemini-3")):
        if any(s in resp_low for s in SMALL_LOCAL_SPOOF_FAMILIES):
            return True
    return False


async def validate_model(session, base, chat_path, model, timeout, retries_429=1, headers=None, extra_payload=None):
    checks, score = {}, 0
    active_model = model
    active_extra = dict(extra_payload) if extra_payload else None

    # Fast Route Gate (Prompt 1): If prompt 1 fails with 4xx or 502/503, try adaptive retries before snapping!
    p1_name, p1_prompt = SOFT_PROMPTS[0]
    res1 = None
    for attempt in range(retries_429 + 1):
        res1 = await chat(session, base, chat_path, active_model, p1_prompt, timeout, headers=headers, extra_payload=active_extra)
        if res1.get("status") == 429 and attempt < retries_429:
            await asyncio.sleep(2.0)
            continue
        break

    st1 = res1.get("status") if res1 else "error"
    err_body = (res1.get("error_body") or "") if isinstance(res1, dict) else ""

    # Vault Lesson #5: NewAPI default-group bypass -> retry with base/<model>
    if st1 in (500, 503) and ("no available channel" in err_body.lower() or "under group" in err_body.lower()) and not active_model.startswith("base/"):
        byp_model = f"base/{active_model}"
        res_byp = await chat(session, base, chat_path, byp_model, p1_prompt, timeout, headers=headers, extra_payload=active_extra)
        if res_byp.get("status") == 200:
            active_model = byp_model
            res1 = res_byp
            st1 = 200
            checks["upstream_id_bypass"] = byp_model

    # Vault Lesson #9: gpt-5/gpt-6/o3 reasoning_effort="none" adaptive retry
    if st1 in (400, 502, 503, "timeout") and any(k in active_model.lower() for k in ("gpt-5", "gpt-6", "o1", "o3", "o4")):
        re_extra = dict(active_extra or {})
        re_extra["reasoning_effort"] = "none"
        res_re = await chat(session, base, chat_path, active_model, p1_prompt, timeout, headers=headers, extra_payload=re_extra)
        if res_re.get("status") == 200:
            active_extra = re_extra
            res1 = res_re
            st1 = 200
            checks["reasoning_effort_none"] = True

    model = active_model
    extra_payload = active_extra

    if st1 == 200:
        res1 = res1 if isinstance(res1, dict) else {}
        c1 = res1.get("content", "")
        raw_c1 = res1.get("raw_content", "") or c1
        if _is_frontier_spoof(model, res1.get("resp_model")):
            checks["model_echo_mismatch"] = f"{model}->{res1.get('resp_model')}"
        if c1.strip() and not check_stub(c1):
            score += 1
            checks[p1_name] = "pass"
            if "<think" in raw_c1.lower() or res1.get("raw_msg", {}).get("reasoning_content"):
                checks["thinking_tag"] = True
                score += 1
        elif res1.get("tool_calls"):
            # 2026-10-09: an injected-toolset gateway answers with a tool call and
            # EMPTY content. That is a live, real host — not a stub. Score partial
            # and record the tool names so the host still surfaces in triage.
            score += 0.5
            checks[p1_name] = "tool_call_only"
            checks["injected_tools"] = res1.get("tool_calls")
        else:
            checks[p1_name] = "stub_or_empty"
        # carry the gateway-behaviour signals up into the per-model checks
        if res1.get("prompt_tokens") is not None:
            checks["prompt_tokens"] = res1.get("prompt_tokens")
        if res1.get("header_leak"):
            checks["header_leak"] = True
        if c1:
            checks["sample"] = c1[:200]
    elif st1 == 429:
        checks[p1_name] = "429_active_relay"
        score += 0.5
    elif isinstance(st1, int) and (400 <= st1 < 500 or st1 in (502, 503)):
        # Early exit: upstream expired key or bad gateway!
        checks[p1_name] = f"fast_snap_{st1}"
        return {"model": model, "score": 0, "tier": f"SNAPPED_{st1}", "checks": checks}
    else:
        checks[p1_name] = f"fail_{st1}"

    # Consecutive-timeout abort: if prompt 1 timed out, try prompt 2.
    # If prompt 2 ALSO times out, snap — this is a tarpit, not slow-but-alive.
    if st1 == "timeout":
        p2_name, p2_prompt = SOFT_PROMPTS[1]
        res2 = await chat(session, base, chat_path, model, p2_prompt, timeout, headers=headers, extra_payload=extra_payload)
        st2 = res2.get("status") if res2 else "error"
        if st2 == "timeout":
            checks["early_abort"] = "consecutive_timeouts"
            return {"model": model, "score": 0, "tier": "TARPIT_TIMEOUT", "checks": checks}
        # Prompt 2 responded — process it through the normal loop below
        # (it will be re-tested as part of the remaining soft prompts)

    # Remaining soft prompts
    for name, prompt in SOFT_PROMPTS[1:]:
        res = await chat(session, base, chat_path, model, prompt, timeout, headers=headers, extra_payload=extra_payload)
        st = res.get("status") if isinstance(res, dict) else "error"
        if st == 200:
            content = res.get("content", "") if isinstance(res, dict) else ""
            if content.strip() and not check_stub(content):
                score += 1
                checks[name] = "pass"
                if name == "logic_riddle":
                    if "7" in content and not ("3" in content and content.index("3") < content.index("7")):
                        score += 1
                        checks["riddle_logic"] = "pass"
                    else:
                        checks["riddle_logic"] = "failed_deduction"
            else:
                checks[name] = "stub_or_empty"
        elif st == 429:
            checks[name] = "429_active_relay"
            score += 0.5
        elif isinstance(st, int) and (400 <= st < 500 or st in (502, 503)):
            checks[name] = f"4xx_{st}"
            return {"model": model, "score": 0, "tier": f"SNAPPED_{st}", "checks": checks}
        else:
            checks[name] = f"fail_{st}"

    # Variance Check (max_tokens=1024 so reasoning models don't starve inside <think>)
    r1 = await chat(session, base, chat_path, model, VARIANCE_PROMPT, timeout, max_tokens=1024, headers=headers, extra_payload=extra_payload)
    r2 = await chat(session, base, chat_path, model, VARIANCE_PROMPT, timeout, max_tokens=1024, headers=headers, extra_payload=extra_payload)
    if r1.get("status") == 200 and r2.get("status") == 200:
        c1, c2 = r1.get("content", "").strip(), r2.get("content", "").strip()
        if c1 and c2 and c1 != c2:
            score += 2
            checks["variance"] = "pass"
        elif c1 and c1 == c2:
            checks["variance"] = "IDENTICAL_REPLAY_SUSPECT"

    # Verifiable Arithmetic Check — word-boundary regex on 391 (max_tokens=1024 for thinking models)
    ra = await chat(session, base, chat_path, model, ARITH_PROMPT, timeout, max_tokens=1024, headers=headers, extra_payload=extra_payload)
    if ra.get("status") == 200:
        content = ra.get("content", "")
        if ARITH_RE.search(content):
            score += 2
            checks["arithmetic"] = "pass"
        else:
            checks["arithmetic"] = f"wrong_answer:{content[:30]!r}"

    # Streaming Fingerprint — with TTFT ceiling and chunk-accuracy (max_tokens=1024)
    rs = await chat(session, base, chat_path, model, STREAM_PROMPT, timeout * 1.5, stream=True, max_tokens=1024, headers=headers, extra_payload=extra_payload)
    if rs.get("status") == 200:
        try:
            chunks = int(rs.get("chunks", 0))
            ttft = float(rs["ttft"]) if rs.get("ttft") is not None else None
            elapsed = float(rs.get("elapsed", 1.0))
        except (ValueError, TypeError):
            chunks, ttft, elapsed = 0, None, 1.0

        if chunks > 3 and ttft is not None and ttft < elapsed * 0.95:
            score += 2
            partial_tag = " (partial)" if rs.get("partial") else ""
            checks["streaming"] = f"pass(ttft={ttft:.2f}s chunks={chunks}{partial_tag})"
        else:
            checks["streaming"] = f"ONE_SHOT_CANNED_SUSPECT(chunks={chunks})"
    elif rs.get("status") == 429:
        checks["streaming"] = "429_active_relay"
        score += 1

    # HARD: Topic-Fidelity Check (max_tokens=1024 so reasoning preamble never starves the haiku!)
    rt = await chat(session, base, chat_path, model, TOPIC_PROMPT, timeout, max_tokens=1024, headers=headers, extra_payload=extra_payload)
    if rt.get("status") == 200:
        ct = rt.get("content", "").lower()
        if any(k in ct for k in TOPIC_KEYWORDS):
            score += 2
            checks["topic_fidelity"] = "pass"
        else:
            checks["topic_fidelity"] = f"CANNED_STUB_FAIL:{ct[:60]!r}"
            return {"model": model, "score": 0, "tier": "FAKE_CANNED_STUB", "checks": checks}
    elif rt.get("status") == 429:
        checks["topic_fidelity"] = "429_active_relay"
        score += 1

    if checks.get("model_echo_mismatch"):
        return {"model": model, "score": score, "tier": "FAKE_MODEL_SPOOF", "checks": checks}
    if checks.get("variance") == "IDENTICAL_REPLAY_SUSPECT" or str(checks.get("arithmetic", "")).startswith("wrong_answer:"):
        return {"model": model, "score": 0, "tier": "FAKE_CANNED_STUB", "checks": checks}

    tier = "GOLD" if score >= 9 else "SILVER" if score >= 6 else "BRONZE" if score >= 3 else "FAKE"
    return {"model": model, "score": score, "tier": tier, "checks": checks}


async def probe_path_fast(session, base, m_path, c_path, headers=None):
    """Fast 3.5s probe to discover model paths in parallel (with default-template-key fallback on 401/403)."""
    try:
        async with asyncio.timeout(3.5):
            async with session.get(f"{base}{m_path}", headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=3.5, connect=CONNECT_TIMEOUT)) as r:
                if r.status == 200:
                    data = await r.json(content_type=None)
                    m_list = []
                    if isinstance(data, dict):
                        raw_m = data.get("data") or data.get("models") or []
                        if isinstance(raw_m, list):
                            for item in raw_m:
                                if isinstance(item, dict):
                                    mid = item.get("id") or item.get("name")
                                    if mid:
                                        s_mid = str(mid)
                                        if s_mid.startswith("models/"):
                                            s_mid = s_mid[len("models/"):]
                                        m_list.append(s_mid)
                                elif isinstance(item, str):
                                    m_list.append(item)
                        elif isinstance(raw_m, dict):
                            m_list.extend([str(k) for k in raw_m.keys()])
                        elif m_path == "/model/info":
                            m_list.extend([str(k) for k in data.keys() if k != "data"])
                    if m_list:
                        return (m_path, c_path, list(dict.fromkeys(m_list)))
                elif r.status in (401, 403) and headers is None and m_path == "/v1/models":
                    for tk in DEFAULT_TEMPLATE_KEYS:
                        hdr = {"Authorization": tk}
                        res_tk = await probe_path_fast(session, base, m_path, c_path, headers=hdr)
                        if res_tk:
                            return (res_tk[0], res_tk[1], res_tk[2], hdr)
    except Exception:
        pass
    return None


PROXY_CANDIDATE_PORTS = {
    1080, 1081, 2080, 3128, 7890, 7891, 8080, 8081, 8082, 8118, 8888, 9050, 9080, 10808, 10809,
}
PURE_PROXY_PORTS = {1080, 1081, 2080, 3128, 7890, 7891, 8118, 9050, 10808, 10809}


async def _probe_socks5_exit(ip: str, port: int, timeout: float = 3.0) -> str | None:
    """Zero-dependency asyncio SOCKS5 no-auth handshake + exit IP check via api.ipify.org:80."""
    writer = None
    try:
        async with asyncio.timeout(timeout):
            reader, writer = await asyncio.open_connection(ip, port)
            writer.write(b"\x05\x01\x00")
            await writer.drain()
            resp = await reader.readexactly(2)
            if resp != b"\x05\x00":
                return None
            target_host = b"api.ipify.org"
            req = b"\x05\x01\x00\x03" + bytes([len(target_host)]) + target_host + (80).to_bytes(2, "big")
            writer.write(req)
            await writer.drain()
            hdr = await reader.readexactly(4)
            if hdr[0] != 5 or hdr[1] != 0:
                return None
            atyp = hdr[3]
            if atyp == 1:
                await reader.readexactly(4 + 2)
            elif atyp == 3:
                dlen = (await reader.readexactly(1))[0]
                await reader.readexactly(dlen + 2)
            elif atyp == 4:
                await reader.readexactly(16 + 2)
            http_req = (
                b"GET /?format=json HTTP/1.1\r\n"
                b"Host: api.ipify.org\r\n"
                b"User-Agent: curl/8.5.0\r\n"
                b"Connection: close\r\n\r\n"
            )
            writer.write(http_req)
            await writer.drain()
            raw = await reader.read(2048)
            txt = raw.decode("utf-8", errors="ignore")
            if "{" in txt and "ip" in txt:
                body = txt[txt.find("{"):txt.rfind("}") + 1]
                data = json.loads(body)
                if isinstance(data, dict) and data.get("ip"):
                    return str(data["ip"])
    except Exception:
        return None
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
    return None


async def _probe_http_connect_tunnel(ip: str, port: int, timeout: float = 2.5) -> bool:
    """Verify the HTTP proxy supports HTTPS CONNECT tunneling (port 443), not just plain HTTP GET."""
    writer = None
    try:
        async with asyncio.timeout(timeout):
            reader, writer = await asyncio.open_connection(ip, port)
            writer.write(b"CONNECT api.ipify.org:443 HTTP/1.1\r\nHost: api.ipify.org:443\r\n\r\n")
            await writer.drain()
            line = await reader.readline()
            return b" 200 " in line
    except Exception:
        return False
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass


import ipaddress

DATACENTER_KEYWORDS = (
    "alibaba", "alicloud", "aliyun", "tencent", "amazon", "aws", "google", "gcp",
    "microsoft", "azure", "digitalocean", "hetzner", "ovh", "choopa", "vultr",
    "linode", "akamai", "cloudflare", "oracle", "contabo", "ucloud", "huawei",
    "kamatera", "leaseweb", "scaleway", "upcloud", "zenlayer", "cogent",
    "hurricane", "datacamp", "m247", "tzulo", "psychz", "multacom", "colocrossing",
    "frantech", "buyvm", "equinix", "hostinger", "ionos", "racknerd",
)


def _is_valid_public_ip(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str.strip())
        return addr.is_global and not addr.is_private and not addr.is_loopback and not addr.is_reserved
    except Exception:
        return False


_RUNNER_OWN_IP = None


async def _get_runner_own_ip(session) -> str:
    global _RUNNER_OWN_IP
    if _RUNNER_OWN_IP is not None:
        return _RUNNER_OWN_IP
    try:
        async with asyncio.timeout(2.5):
            async with session.get("http://api.ipify.org/?format=json", timeout=aiohttp.ClientTimeout(total=2.5)) as r:
                if r.status == 200:
                    d = await r.json(content_type=None)
                    if isinstance(d, dict) and d.get("ip"):
                        _RUNNER_OWN_IP = str(d["ip"]).strip()
                        return _RUNNER_OWN_IP
    except Exception:
        pass
    _RUNNER_OWN_IP = ""
    return _RUNNER_OWN_IP


async def _classify_exit_ip(session, scanned_ip: str, exit_ip: str, proto: str, https_connect: bool) -> dict | None:
    """Check whether exit_ip belongs to a residential/mobile ISP vs datacenter hosting.
    Requires a valid global exit_ip and HTTPS CONNECT support for HTTP proxies."""
    if not _is_valid_public_ip(exit_ip):
        return None
    if proto == "http" and not https_connect:
        return None
    own_ip = await _get_runner_own_ip(session)
    if own_ip and exit_ip == own_ip:
        return None
    relayed = (exit_ip != scanned_ip)
    info = {
        "sidecar": "OPEN_FORWARD_PROXY",
        "protocol": proto,
        "https_connect": https_connect,
        "scanned_ip": scanned_ip,
        "exit_ip": exit_ip,
        "relayed_exit": relayed,
    }
    classified = False
    try:
        async with asyncio.timeout(2.5):
            url = f"http://ip-api.com/json/{exit_ip}?fields=status,countryCode,isp,org,as,mobile,proxy,hosting"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.5)) as r:
                if r.status == 200:
                    meta = await r.json(content_type=None)
                    if isinstance(meta, dict) and meta.get("status") == "success":
                        is_hosting = bool(meta.get("hosting"))
                        is_mobile = bool(meta.get("mobile"))
                        isp_org = f"{meta.get('isp', '')} {meta.get('org', '')} {meta.get('as', '')}".lower()
                        if any(k in isp_org for k in DATACENTER_KEYWORDS):
                            is_hosting = True
                        info["country"] = meta.get("countryCode")
                        info["isp"] = meta.get("isp")
                        info["asn"] = meta.get("as")
                        info["hosting"] = is_hosting
                        info["mobile"] = is_mobile
                        info["sidecar"] = "OPEN_RESIDENTIAL_PROXY" if (not is_hosting or is_mobile) else "OPEN_DATACENTER_PROXY"
                        classified = True
    except Exception:
        pass

    # Fallback to ipwho.is if ip-api.com rate-limited (45 RPM cap)
    if not classified:
        try:
            async with asyncio.timeout(2.5):
                async with session.get(f"https://ipwho.is/{exit_ip}", timeout=aiohttp.ClientTimeout(total=2.5)) as r2:
                    if r2.status == 200:
                        m2 = await r2.json(content_type=None)
                        if isinstance(m2, dict) and m2.get("success") is True:
                            raw_conn = m2.get("connection")
                            conn_info = raw_conn if isinstance(raw_conn, dict) else {}
                            isp_org = f"{conn_info.get('isp', '')} {conn_info.get('org', '')} {conn_info.get('asn', '')}".lower()
                            is_dc = any(k in isp_org for k in DATACENTER_KEYWORDS)
                            info["country"] = m2.get("country_code")
                            info["isp"] = conn_info.get("isp")
                            info["asn"] = str(conn_info.get("asn", ""))
                            info["hosting"] = is_dc
                            info["sidecar"] = "OPEN_DATACENTER_PROXY" if is_dc else "OPEN_RESIDENTIAL_PROXY"
        except Exception:
            pass
    return info


async def _sidecar_triage(session, ip: str, port: int, base: str) -> dict | None:
    """Sidecar triage for non-LLM high-value infra:
    1. Port 2375 Docker 5-point read-side authenticity gate
    2. Open-signup OneAPI/NewAPI (/api/status) & Open-WebUI (/api/config)
    3. SOCKS5 + HTTP CONNECT egress proxy check (classifying Residential vs Datacenter exit IP)
    """
    if port == 2375:
        try:
            async with asyncio.timeout(4.0):
                async with session.get(f"{base}/_ping", timeout=aiohttp.ClientTimeout(total=2.5)) as rp:
                    ping_txt = (await rp.text()).strip()
                    srv = rp.headers.get("Server", "")
                    if rp.status != 200 or ping_txt != "OK" or not srv.startswith("Docker/"):
                        return None
                async with session.get(f"{base}/version", timeout=aiohttp.ClientTimeout(total=2.5)) as rv1:
                    v1 = await rv1.json(content_type=None)
                async with session.get(f"{base}/version", timeout=aiohttp.ClientTimeout(total=2.5)) as rv2:
                    v2 = await rv2.json(content_type=None)
                if not isinstance(v1, dict) or not {"Version", "ApiVersion", "Os", "Arch"}.issubset(v1.keys()):
                    return None
                if v1.get("Version") != v2.get("Version") or v1.get("GitCommit") != v2.get("GitCommit"):
                    return None
                async with session.get(f"{base}/nonexistent_9f8a7b", timeout=aiohttp.ClientTimeout(total=2.5)) as rb:
                    if rb.status != 404:
                        return None
                return {"sidecar": "DOCKER_DAEMON_AUTHENTIC", "version": v1.get("Version")}
        except Exception:
            return None

    if port not in PURE_PROXY_PORTS:
        for p_path, p_kind in (("/api/status", "newapi"), ("/api/config", "open_webui")):
            try:
                async with asyncio.timeout(2.5):
                    async with session.get(f"{base}{p_path}", timeout=aiohttp.ClientTimeout(total=2.5)) as r:
                        if r.status == 200:
                            d = await r.json(content_type=None)
                            if isinstance(d, dict):
                                raw_inner = d.get("data")
                                inner = raw_inner if isinstance(raw_inner, dict) else d
                                if p_kind == "newapi" and inner.get("register_enabled") is True and not inner.get("turnstile_check"):
                                    return {"sidecar": "OPEN_SIGNUP_NEWAPI", "system_name": inner.get("system_name")}
                                if p_kind == "open_webui" and inner.get("enable_signup") is True:
                                    return {"sidecar": "OPEN_SIGNUP_WEBUI", "name": inner.get("name")}
            except Exception:
                pass

    if port in PROXY_CANDIDATE_PORTS:
        # 1. Try SOCKS5 no-auth tunnel first on SOCKS/Mixed ports (Clash 7890, v2ray 10808, Sing-box 2080, SOCKS 1080)
        if port in (1080, 1081, 2080, 7890, 7891, 9050, 10808):
            s5_exit = await _probe_socks5_exit(ip, port, timeout=3.0)
            if s5_exit:
                return await _classify_exit_ip(session, ip, s5_exit, proto="socks5", https_connect=True)
        # 2. Try HTTP forward proxy + verify HTTPS CONNECT (port 443)
        try:
            async with asyncio.timeout(3.0):
                async with session.get("http://api.ipify.org/?format=json", proxy=f"http://{ip}:{port}",
                                       timeout=aiohttp.ClientTimeout(total=3.0, connect=1.5)) as rp:
                    if rp.status == 200:
                        pd = await rp.json(content_type=None)
                        if isinstance(pd, dict) and pd.get("ip"):
                            can_connect = await _probe_http_connect_tunnel(ip, port, timeout=2.5)
                            return await _classify_exit_ip(session, ip, str(pd["ip"]), proto="http", https_connect=can_connect)
        except Exception:
            pass
    return None


async def validate_host(session, host, timeout, max_models, sem, ledger_fp=None, ledger_lock=None):
    if ":" not in host:
        result = {"host": host, "tier": "DEAD", "error": "invalid_format", "working_models": []}
        _ledger_write(ledger_fp, ledger_lock, result)
        return result

    ip, port_str = host.rsplit(":", 1)
    try:
        port = int(port_str)
    except ValueError:
        result = {"host": host, "tier": "DEAD", "error": "invalid_port", "working_models": []}
        _ledger_write(ledger_fp, ledger_lock, result)
        return result

    # Zero-Ollama Guard (bypass with --include-ollama)
    if port == 11434:
        result = {"host": host, "tier": "FILTERED_OLLAMA", "error": "ollama_excluded", "working_models": []}
        _ledger_write(ledger_fp, ledger_lock, result)
        return result

    scheme = "https" if port in (443, 8443, 2053, 2083, 2087, 2096) else "http"
    base = f"{scheme}://{ip}:{port}"
    host_t0 = time.time()
    async with sem:
        # Fast-path pure proxy/tunnel ports (1080, 7890, 10808, 3128, etc.) straight to proxy triage
        if port in PURE_PROXY_PORTS:
            sidecar = await _sidecar_triage(session, ip, port, base)
            result = {"host": host, "tier": sidecar["sidecar"] if sidecar else "DEAD",
                      "error": "pure_proxy_port", "sidecar": sidecar, "working_models": []}
            _ledger_write(ledger_fp, ledger_lock, result)
            return result

        # Step 1: Probe all candidate paths simultaneously via asyncio.gather!
        path_tasks = [probe_path_fast(session, base, m_p, c_p) for m_p, c_p in CANDIDATE_PATHS]
        path_results = await asyncio.gather(*path_tasks)

        discovered = next((res for res in path_results if res), None)

        # TLS Fallback: if plain HTTP failed on a non-443 port, try https:// if port is 8443/443 or dual-stack
        if not discovered and scheme == "http" and port in (3000, 3001, 4000, 8000, 8080, 8443):
            https_base = f"https://{ip}:{port}"
            https_results = await asyncio.gather(*(probe_path_fast(session, https_base, m_p, c_p) for m_p, c_p in CANDIDATE_PATHS[:2]))
            https_disc = next((res for res in https_results if res), None)
            if https_disc:
                base = https_base
                discovered = https_disc

        if not discovered:
            sidecar = await _sidecar_triage(session, ip, port, base)
            result = {"host": host, "tier": sidecar["sidecar"] if sidecar else "DEAD",
                      "error": "no_models_endpoint_responsive", "sidecar": sidecar, "working_models": []}
            _ledger_write(ledger_fp, ledger_lock, result)
            return result

        if len(discovered) == 4:
            discovered_path, discovered_chat, raw_models, auth_headers = discovered
        else:
            discovered_path, discovered_chat, raw_models = discovered
            auth_headers = None

        # Filter to chat models only and priority-sort frontier/tea-channel aliases first!
        chat_models = [m for m in raw_models if is_chat_model(m)]
        if not chat_models:
            chat_models = raw_models  # fallback if all got filtered
        chat_models = sort_models_by_priority(chat_models)

        # Cap generative models if specified (default 25)
        models_to_test = chat_models if max_models <= 0 else chat_models[:max_models]

        # Step 2: Validate models concurrently (Semaphore(2) to avoid self-inflicted RPM 429s)
        model_sem = asyncio.Semaphore(2)
        async def run_m(m):
            async with model_sem:
                return await validate_model(session, base, discovered_chat, m, timeout, headers=auth_headers)

        # Host-level total budget — collect partial results even on timeout
        results = []
        try:
            async with asyncio.timeout(HOST_BUDGET):
                async def collect(m):
                    r = await run_m(m)
                    results.append(r)
                    return r
                await asyncio.gather(*(collect(m) for m in models_to_test))
        except asyncio.TimeoutError:
            # Budget exhausted — fill remaining with placeholder
            for m in models_to_test:
                if not any(r.get("model") == m for r in results):
                    results.append({"model": m, "score": 0, "tier": "HOST_BUDGET_EXHAUSTED", "checks": {}})

        working = [r for r in results if r["tier"] in ("GOLD", "SILVER")]
        snapped = [r for r in results if r["tier"].startswith("SNAPPED_")]

        if any(r["tier"] == "GOLD" for r in results):
            host_tier = "GOLD"
        elif working:
            host_tier = "SILVER"
        elif snapped and len(snapped) == len(results):
            host_tier = "SNAPPED_4XX_HOST"
        else:
            host_tier = "FAKE"

        # 2026-10-09: gateway-behaviour signals. These are what the payload triage
        # found on the wired fleet — injected system prompts, injected toolsets,
        # injected personas and owner-ID leaks. Capture them at discovery time so a
        # suspicious host is FLAGGED rather than silently wired.
        sig_tools, sig_sys, sig_leak, sig_persona = set(), [], False, []
        for r in results:
            ch = r.get("checks") or {}
            for t in (ch.get("injected_tools") or []):
                sig_tools.add(t)
            pt = ch.get("prompt_tokens")
            if isinstance(pt, int) and pt > 60:
                sig_sys.append(pt)
        for r in results:
            ch = r.get("checks") or {}
            if ch.get("header_leak"):
                sig_leak = True
        joined = " ".join(
            str((r.get("checks") or {}).get("sample") or "") for r in results
        ).lower()
        for mark in INJECTED_PERSONA_MARKERS:
            if mark in joined:
                sig_persona.append(mark)
        gateway_signals = {
            "injected_tools": sorted(sig_tools)[:8],
            "sysprompt_tokens": max(sig_sys) if sig_sys else None,
            "header_leak": sig_leak,
            "injected_persona": sig_persona,
        }

        out_obj = {
            "host": host,
            "ip": ip,
            "port": port,
            "endpoint_path": discovered_path,
            "chat_path": discovered_chat,
            "tier": host_tier,
            "total_models": len(raw_models),
            "tested_models_count": len(results),
            "working_models_count": len(working),
            "working_models": working,
            "all_results": results,
            "gateway_signals": gateway_signals,
            "elapsed_s": round(time.time() - host_t0, 2)
        }

        # Real-time stdout logging!
        if host_tier in ("GOLD", "SILVER"):
            flag = "👑" if host_tier == "GOLD" else "🥈"
            print(f"{flag} {host:22} {host_tier:8} working={len(working)}/{len(raw_models)} ({[m['model'] for m in working][:3]})", flush=True)

        _ledger_write(ledger_fp, ledger_lock, out_obj)
        return out_obj


def _ledger_write(ledger_fp, ledger_lock, record):
    """Append one host result to the JSONL ledger immediately (atomic per record)."""
    if ledger_fp is None:
        return
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if ledger_lock:
        ledger_lock.acquire()
    try:
        ledger_fp.write(line)
        ledger_fp.flush()
    finally:
        if ledger_lock:
            ledger_lock.release()


async def main():
    ap = argparse.ArgumentParser(description="Turbo Hardened Multi-Model Router Validator")
    ap.add_argument("--hosts", required=True, help="Comma-separated host:port list, path to hosts file, or path to masscan -oJ JSON")
    ap.add_argument("--models", type=int, default=25, help="Max models to test per host (0=ALL, default: 25)")
    ap.add_argument("--workers", type=int, default=32, help="Concurrent host workers (default: 32)")
    ap.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout seconds")
    ap.add_argument("--include-ollama", action="store_true", help="Include port 11434 (default: filtered)")
    ap.add_argument("--force-bare-host", action="store_true", help="Explicitly allow running directly on bare host (DANGEROUS)")
    ap.add_argument("-o", "--out", default="validate_v2_results.json", help="Output JSON path")
    args = ap.parse_args()

    # Operator Egress Guard: Never run bulk scan validation on bare residential host!
    if not os.environ.get("GITHUB_ACTIONS") and not args.force_bare_host:
        sys.exit(
            "🚨 FATAL OPERATOR SAFETY VIOLATION: validate_v2.py must run on GitHub Actions hosted runners,\n"
            "NOT directly on the bare residential host! Running this locally burns the operator's home IP.\n"
            "If you intentionally want to test a single host locally, pass --force-bare-host."
        )

    hosts_input = args.hosts.strip()

    # Universal loader: masscan -oJ JSON array, line-delimited ip:port, or comma-separated
    if os.path.isfile(hosts_input):
        raw = open(hosts_input).read().strip()
        if raw.startswith("["):
            # masscan -oJ format
            hits = json.loads(raw)
            hosts = []
            for h in hits:
                ip = h.get("ip", "")
                for p in h.get("ports", []):
                    pnum = p.get("port") if isinstance(p, dict) else p
                    if pnum:
                        hosts.append(f"{ip}:{pnum}")
        else:
            hosts = [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("#")]
    else:
        hosts = [h.strip() for h in hosts_input.replace("\n", ",").split(",") if h.strip()]

    # Filter out Ollama port 11434 unless --include-ollama
    if args.include_ollama:
        filtered_hosts = hosts
    else:
        filtered_hosts = [h for h in hosts if not h.endswith(":11434")]
    dropped_ollama = len(hosts) - len(filtered_hosts)
    if dropped_ollama > 0:
        print(f"🛡️ Pre-filtered {dropped_ollama} Ollama hosts (:11434). Remaining: {len(filtered_hosts)}", flush=True)

    print(f"🚀 Starting Turbo Validator on {len(filtered_hosts)} hosts with {args.workers} workers (models_cap={args.models})...", flush=True)
    t0 = time.time()

    # JSONL ledger for incremental results
    ledger_path = args.out.replace(".json", ".jsonl") if args.out.endswith(".json") else args.out + ".jsonl"
    ledger_fp = open(ledger_path, "a")
    import threading
    ledger_lock = threading.Lock()

    sem = asyncio.Semaphore(args.workers)
    conn = aiohttp.TCPConnector(limit=args.workers * 4, ttl_dns_cache=300, ssl=False)
    async with aiohttp.ClientSession(connector=conn) as session:
        out = await asyncio.gather(*(validate_host(session, h, args.timeout, args.models, sem, ledger_fp, ledger_lock) for h in filtered_hosts))

    elapsed = time.time() - t0
    ledger_fp.close()
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2)

    from collections import Counter
    counts = dict(Counter(r["tier"] for r in out))
    print(f"\n✨ Completed in {elapsed:.1f}s ({len(filtered_hosts)/max(elapsed, 0.1):.1f} hosts/s)")
    print(f"📊 Host Tiers Summary: {counts}", flush=True)
    print(f"📋 Incremental ledger: {ledger_path}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
