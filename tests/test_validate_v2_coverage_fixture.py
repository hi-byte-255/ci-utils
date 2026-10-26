"""Loopback fixture: end-to-end proof of the Y6 coverage/resume path against a
dense-but-correct synthetic router. Binds 127.0.0.1 only — zero external egress.

The fixture advertises 11 chat models, answers every battery prompt correctly, streams
SSE for the streaming probe, and returns EMPTY prose for exactly one model (the A1
alive-but-empty class). Two passes are then run against the same --out ledger:
  pass 1 with a starved HOST_BUDGET -> partial coverage + HOST_BUDGET_EXHAUSTED tail
  pass 2 -> prior-ALIVE rows are reused (reused_from_ledger > 0) and coverage reaches 1.0
"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

MODELS = [
    "claude-opus-4-6", "gpt-5-mini", "gemini-3-pro", "deepseek-v4", "qwen3-235b-instruct",
    "claude-sonnet-4-6", "gpt-6", "glm-5", "kernel-k3", "muse-glimmer-30b", "empty-prose-model",
]
EMPTY_MODEL = "empty-prose-model"
SLOW_S = 0.02  # per-request latency so the starved pass is deterministic


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # keep the test output clean
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the validator cancelled the call (host budget) — nothing to answer

    def _sse(self, model):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(6):
                chunk = {"choices": [{"delta": {"content": str(i)}}], "model": model}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.005)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._json({"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}
        prompt = ((req.get("messages") or [{}])[0] or {}).get("content", "") or ""
        model = req.get("model") or ""

        if "17*23" in prompt:
            content = "391"
        elif "haiku" in prompt:
            content = "" if model == EMPTY_MODEL else (
                "teal bicycle by the pier\nwheels hum over wet concrete\nthe chain sings all the way home")
        elif "ocean at night" in prompt:
            content = f"the tide keeps its own ledger, entry {time.monotonic_ns()}"
        elif "fizzbuzz" in prompt:
            content = ("def fizzbuzz(n):\n"
                       "    return 'FizzBuzz' if n % 15 == 0 else 'Fizz' if n % 3 == 0 "
                       "else 'Buzz' if n % 5 == 0 else str(n)")
        elif "sheep" in prompt:
            content = "7"
        elif "palindrome" in prompt:
            content = "def is_palindrome(s):\n    return s == s[::-1]"
        else:
            content = f"a still thought about deep water, variant {time.monotonic_ns()}"

        if req.get("stream"):
            self._sse(model)
            return

        time.sleep(SLOW_S)
        self._json({
            "id": "fixture", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 8, "total_tokens": 29},
        })


@pytest.fixture()
def router():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _run(host, out_path, budget_s, models=0):
    env = dict(os.environ, HOST_BUDGET=str(budget_s), MODEL_SEM_START="2", MODEL_SEM_MAX="6")
    proc = subprocess.run(
        [PY, os.path.join(REPO, "validate_v2.py"), "--hosts", host, "--models", str(models),
         "--workers", "4", "--timeout", "6", "--force-bare-host", "-o", str(out_path)],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"validator exited {proc.returncode}\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}"
    with open(out_path) as fp:
        return json.load(fp)[0], proc.stdout


def test_full_coverage_and_empty_content_class(router, tmp_path):
    rec, log = _run(router, tmp_path / "validated.json", 120)
    assert rec["advertised_chat"] == len(MODELS), rec.get("advertised_chat")
    assert rec["coverage"] == 1.0, rec.get("coverage")
    assert rec["budget_exhausted"] == 0, rec.get("budget_exhausted")
    assert rec["tier"] == "GOLD", rec.get("tier")
    assert rec["elapsed_s"] >= SLOW_S

    by_model = {r["model"]: r["tier"] for r in rec["all_results"]}
    assert by_model[EMPTY_MODEL] == "EMPTY_CONTENT_ALIVE", by_model
    assert by_model["claude-opus-4-6"] in ("GOLD", "SILVER"), by_model
    assert EMPTY_MODEL not in [m["model"] for m in rec["working_models"]]
    assert any(p["model"] == EMPTY_MODEL for p in rec["partial_models"])
    assert "coverage=11/11" in log


def test_partial_pass_then_resume_converges(router, tmp_path):
    """A partial first pass (model cap stands in for a starved host budget) must be
    completed by a second pass that REUSES the prior-ALIVE rows instead of re-testing
    them. The cap variant is used because it exercises the identical resume path
    (resume_plan + ledger reload) without depending on wall-clock timing."""
    out = tmp_path / "validated.json"
    rec1, log1 = _run(router, out, 120, models=3)
    assert rec1["advertised_chat"] == len(MODELS)
    assert rec1["coverage"] < 1.0, rec1.get("coverage")
    assert rec1["reused_from_ledger"] == 0
    alive1 = [r["model"] for r in rec1["all_results"] if r["tier"] in ("GOLD", "SILVER")]
    assert alive1, "first pass produced no ALIVE models to carry forward"
    assert len(alive1) == 3, alive1

    rec2, log2 = _run(router, out, 120, models=0)      # same ledger => resume
    assert rec2["reused_from_ledger"] == len(alive1), rec2.get("reused_from_ledger")
    assert rec2["coverage"] == 1.0, rec2.get("coverage")
    assert "Resume:" in log2
    verdicts = {r["model"]: r["tier"] for r in rec2["all_results"]}
    assert set(verdicts) == set(MODELS), verdicts
    assert all(verdicts[m] in ("GOLD", "SILVER") for m in alive1), verdicts
    assert verdicts[EMPTY_MODEL] == "EMPTY_CONTENT_ALIVE", verdicts
