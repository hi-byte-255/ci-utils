#!/usr/bin/env python3
"""Wave merge + middlebox filter (pipeline stage 1).

masscan ``-oJ`` emits ``{ip, ports:[{port,status}], timestamp}`` per record with
a trailing comma before the final ``]`` (invalid JSON) — tolerated here.

Stage 1 turns raw shard hits into a deduped ``ip -> {open ports}`` map and then
splits it:

  * **middleboxes** — hosts that answer SYN on almost *every* scanned port
    (VPN concentrators, firewalls, hosting panels). They are noise: they never
    speak the OpenAI protocol, and probing them wasted hours last wave.
  * **candidates**  — hosts with a small, deliberate set of open ports. This is
    where real gateways live (verified wave-3 gateways sat on 1-3 ports).

Wave 3 shipped the comparison inverted (``len(ps) >= MIN_PORTS``), so stage 2
probed 442k SYN-ACK firewalls for 2.6 h while the real servers were never
checked. The predicate below is corrected and pinned by
``tests/test_merge_filter.py``.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

HITS_DIR = os.environ.get("HITS_DIR", "./hits")
OUT_DIR = os.environ.get("OUT_DIR", ".")

# Hosts answering on >= this many of our scanned ports are middleboxes.
MIDDLEBOX_PORTS = int(os.environ.get("MIDDLEBOX_PORTS", "10"))

# Probe order for stage 2: most gateway-dense ports first (wave-3 + batch25 yield + proxy/tunnel ports).
PORT_PRIORITY = [
    8080, 3000, 8000, 5000, 8081, 5001, 8001, 8002, 8005, 8082, 8085, 8086,
    8088, 8089, 8090, 8100, 8125, 8126, 3001, 3002, 3003, 3004, 3005, 4000,
    30000, 1234, 18081, 18082, 23333, 52444, 8443, 443, 80, 11434, 7860,
    7861, 2718, 5002, 9000, 9001, 9200, 5678, 8501, 8888, 6333, 2375,
    7890, 7891, 10808, 10809, 1080, 2080, 3128, 8118,
]

# High-value LLM/router/infra/proxy ports that should ALL be emitted when open on a clean IP
MULTI_EMIT_PORTS = {
    3000, 3001, 3002, 3003, 3004, 3005, 4000, 5000, 5001, 5002,
    7860, 7861, 8000, 8001, 8002, 8005, 8080, 8081, 8082, 8085,
    8086, 8088, 8089, 8090, 8100, 8125, 8126, 8443, 8888, 9000,
    9001, 1234, 18081, 18082, 23333, 30000, 52444, 2375,
    7890, 7891, 10808, 10809, 1080, 2080, 3128, 8118,
}


def load_hits(path: str) -> list:
    """Parse a masscan -oJ file, tolerating the trailing-comma quirk."""
    txt = open(path, encoding="utf-8", errors="ignore").read().strip()
    if not txt or txt in ("[]",):
        return []
    if txt.endswith(","):
        txt = txt[:-1]
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        txt = txt.rstrip().rstrip(",")
        if not txt.endswith("]"):
            txt += "]"
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return []


def collect_ip_ports(files) -> dict:
    """Merge shard hit files into {ip: {open ports}}."""
    ip_ports: dict = defaultdict(set)
    for f in files:
        for r in load_hits(f):
            ip = r.get("ip")
            if not ip:
                continue
            for p in r.get("ports") or []:
                if p.get("status") == "open":
                    ip_ports[ip].add(p.get("port"))
    return ip_ports


def split_candidates(ip_ports: dict, middlebox_ports: int = MIDDLEBOX_PORTS):
    """Split {ip: ports} into (candidates, middleboxes).

    Candidates are hosts whose open-port count is *below* the middlebox
    threshold — the deliberate servers worth probing.
    """
    candidates, middleboxes = {}, {}
    for ip, ps in ip_ports.items():
        bucket = middleboxes if len(ps) >= middlebox_ports else candidates
        bucket[ip] = set(ps)
    return candidates, middleboxes


def pick_probe_port(ports) -> int:
    """Choose the single best port to probe for a host."""
    available = set(ports)
    for p in PORT_PRIORITY:
        if p in available:
            return p
    return min(available)


def pick_probe_ports(ports) -> list[int]:
    """Return all high-yield LLM/router ports open on a candidate host (ordered by priority),
    falling back to [pick_probe_port(ports)] if only generic ports (e.g. 80/443) are open."""
    available = set(ports)
    ordered_hv = [p for p in PORT_PRIORITY if p in available and p in MULTI_EMIT_PORTS]
    if ordered_hv:
        return ordered_hv
    return [pick_probe_port(available)]


def main() -> int:
    files = sorted(glob.glob(f"{HITS_DIR}/**/hits_*.json", recursive=True))
    if not files:
        print(f"no shard files found under {HITS_DIR} — download first")
        return 1
    print(f"merging {len(files)} shard files (middlebox_ports={MIDDLEBOX_PORTS})")

    ip_ports = collect_ip_ports(files)
    total_pairs = sum(len(v) for v in ip_ports.values())
    print(f"unique hosts       : {len(ip_ports):,}")
    print(f"open ip:port pairs : {total_pairs:,}")

    candidates, middleboxes = split_candidates(ip_ports)
    cand_pairs = sum(len(v) for v in candidates.values())
    dropped_pct = len(middleboxes) / max(1, len(ip_ports)) * 100
    print(f"candidates (<{MIDDLEBOX_PORTS} ports) : {len(candidates):,} hosts ({cand_pairs:,} pairs)")
    print(f"middleboxes (>={MIDDLEBOX_PORTS} ports): {len(middleboxes):,} hosts ({dropped_pct:.1f}%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/candidates_all.jsonl", "w") as fh:
        for ip, ps in sorted(ip_ports.items()):
            fh.write(json.dumps({"ip": ip, "port_count": len(ps),
                                 "ports": sorted(ps)}) + "\n")
    with open(f"{OUT_DIR}/candidates_clean.jsonl", "w") as fh:
        for ip, ps in sorted(candidates.items(), key=lambda kv: -len(kv[1])):
            fh.write(json.dumps({"ip": ip, "port_count": len(ps),
                                 "ports": sorted(ps)}) + "\n")
    with open(f"{OUT_DIR}/middleboxes.jsonl", "w") as fh:
        for ip, ps in sorted(middleboxes.items(), key=lambda kv: -len(kv[1])):
            fh.write(json.dumps({"ip": ip, "port_count": len(ps),
                                 "ports": sorted(ps)}) + "\n")
    # Emit all high-yield LLM/router ports per candidate host -> feeds validate_v2.py
    with open(f"{OUT_DIR}/hosts_pass1.txt", "w") as fh:
        for ip, ps in candidates.items():
            for p in pick_probe_ports(ps):
                fh.write(f"{ip}:{p}\n")

    print(f"\nwrote {OUT_DIR}/candidates_clean.jsonl  <- real servers")
    print(f"wrote {OUT_DIR}/middleboxes.jsonl      <- dropped noise")
    print(f"wrote {OUT_DIR}/hosts_pass1.txt        <- feed this to validate_v2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
