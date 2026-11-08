"""Regression tests for the 2026-10-26 runner-side validator changes (Y6 + A1).

Covers:
  * A1  alive-but-empty content is a PARTIAL class, never a canned-stub kill
  * A1  a genuine topic miss is still a hard kill (previously-correct kills unchanged)
  * A1  the richer retry recovers a model whose prose comes back
  * Y6  resume_plan carries prior-ALIVE models forward and always re-queues failures
  * Y6  AdaptiveSem widens only on fast, 429-free evidence and respects its cap
  * Y6  the host record exposes the coverage evidence AC-1 asks for
"""
import asyncio
import inspect
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import validate_v2 as v  # noqa: E402


def _chat_factory(topic_content, counter=None):
    counter = counter if counter is not None else {"n": 0}

    async def fake_chat(session, base, chat_path, model, prompt, timeout, stream=False,
                        max_tokens=2048, headers=None, extra_payload=None):
        if prompt == v.TOPIC_PROMPT:
            return {"status": 200, "content": topic_content}
        if prompt == v.ARITH_PROMPT:
            return {"status": 200, "content": "391"}
        if stream:
            return {"status": 200, "chunks": 6, "ttft": 0.3, "elapsed": 1.0}
        counter["n"] += 1
        return {"status": 200, "content": f"The tide writes silver notes, line {counter['n']}."}

    return fake_chat


def _run_model(monkeypatch, topic_content):
    monkeypatch.setattr(v, "chat", _chat_factory(topic_content))
    return asyncio.run(v.validate_model(None, "http://x", "/v1/chat/completions", "m", 5.0))


def test_empty_content_is_alive_partial_not_canned_stub(monkeypatch):
    r = _run_model(monkeypatch, "")
    assert r["tier"] == "EMPTY_CONTENT_ALIVE", r
    assert r["checks"]["topic_fidelity"] == "EMPTY_CONTENT_ALIVE"
    assert r["checks"].get("empty_content_retry") is False
    assert r["score"] >= 9, r          # it scored like a strong model ...
    assert r["tier"] not in ("GOLD", "SILVER")   # ... but is never wired as capacity


def test_topic_miss_is_still_a_hard_kill(monkeypatch):
    r = _run_model(monkeypatch, "the sea is calm and grey tonight")
    assert r["tier"] == "FAKE_CANNED_STUB", r
    assert r["checks"]["topic_fidelity"].startswith("CANNED_STUB_FAIL")


def test_empty_content_retry_recovers_when_prose_returns(monkeypatch):
    state = {"topic_calls": 0, "n": 0}

    async def fake_chat(session, base, chat_path, model, prompt, timeout, stream=False,
                        max_tokens=2048, headers=None, extra_payload=None):
        if prompt == v.TOPIC_PROMPT:
            state["topic_calls"] += 1
            if state["topic_calls"] == 1:
                return {"status": 200, "content": ""}
            assert extra_payload and extra_payload.get("reasoning_effort") == "none"
            return {"status": 200, "content": "a teal bicycle leans / wheels spin past the quiet glass"}
        if prompt == v.ARITH_PROMPT:
            return {"status": 200, "content": "391"}
        if stream:
            return {"status": 200, "chunks": 6, "ttft": 0.3, "elapsed": 1.0}
        state["n"] += 1
        return {"status": 200, "content": f"Silver notes on water, take {state['n']}."}

    monkeypatch.setattr(v, "chat", fake_chat)
    r = asyncio.run(v.validate_model(None, "http://x", "/v1/chat/completions", "m", 5.0))
    assert r["checks"]["topic_fidelity"] == "pass", r
    assert r["tier"] in ("GOLD", "SILVER"), r


def test_resume_plan_carries_alive_and_retries_failures():
    prior = {"a": {"model": "a", "tier": "GOLD"}, "z": {"model": "z", "tier": "GOLD"}}
    reused, todo = v.resume_plan(["a", "b", "c"], prior, 0)
    assert [r["model"] for r in reused] == ["a"]
    assert todo == ["b", "c"]                    # failures/unknowns are re-tested
    reused, todo = v.resume_plan(["a", "b", "c"], prior, 1)
    assert todo == ["b"]                         # cap applies to the REMAINING tail
    reused, todo = v.resume_plan(["a", "b"], {}, 0)
    assert reused == [] and todo == ["a", "b"]


def test_adaptive_sem_widens_on_fast_and_holds_under_429():
    sem = v.AdaptiveSem(start=2, cap=4, fast_s=0.35, min_samples=2)
    sem.observe(0.1)
    sem.observe(0.1)
    assert sem.cur == 3
    sem.observe(0.1)
    sem.observe(0.1)
    assert sem.cur == 4
    sem.observe(0.1)
    sem.observe(0.1)
    assert sem.cur == 4                          # cap holds

    slow = v.AdaptiveSem(start=2, cap=6, fast_s=0.35, min_samples=2)
    for _ in range(4):
        slow.observe(2.0)
    assert slow.cur == 2                         # a slow host never widens

    # An early 429 must veto widening entirely: the streak resets and the bar becomes
    # _fast > 3 * throttled (4 consecutive fast samples), so a throttled host stays narrow.
    throttled = v.AdaptiveSem(start=2, cap=6, fast_s=0.35, min_samples=2)
    throttled.observe(0.1)
    throttled.observe(1.0, hit_429=True)
    throttled.observe(0.1)
    throttled.observe(0.1)
    assert throttled.cur == 2

    # A 429 after widening does not shrink the live window (asyncio.Semaphore cannot
    # safely reclaim a permit that a task is holding); it only blocks further widening.
    widened = v.AdaptiveSem(start=2, cap=6, fast_s=0.35, min_samples=2)
    widened.observe(0.1)
    widened.observe(0.1)
    assert widened.cur == 3
    widened.observe(0.5, hit_429=True)
    widened.observe(0.1)
    widened.observe(0.1)
    assert widened.cur == 3


def test_visible_text_falls_back_to_reasoning_channel():
    assert v._visible_text({"content": ""}) == ""
    assert v._visible_text({"content": "", "raw_msg": {"reasoning_content": "think"}}) == "think"
    assert v._visible_text({"content": "", "reasoning_content": "think2"}) == "think2"
    assert v._visible_text(None) == ""


def test_host_record_exposes_coverage_evidence():
    sig = inspect.signature(v.validate_host)
    assert "prior_alive" in sig.parameters
    src = inspect.getsource(v.validate_host)
    for key in ("advertised_chat", "tested_model_ids", "reused_from_ledger",
                "budget_exhausted", "coverage", "partial_models"):
        assert key in src, key


def test_cross_vendor_model_echo_and_pretender_spoof_detection():
    """A router asked for claude-opus-4-6 that echoes gpt-4o-mini or qwen-plus in resp['model'],
    or self-IDs as Qwen/DeepSeek in prose, must be caught as a spoof."""
    assert v._is_frontier_spoof("claude-opus-4-6", "gpt-4o-mini") is True
    assert v._is_frontier_spoof("claude-sonnet-4-6", "qwen-plus-latest") is True
    assert v._is_frontier_spoof("gpt-5.4-pro", "deepseek-chat") is True
    assert v._is_frontier_spoof("gemini-2.5-pro", "glm-4-flash") is True
    # Legitimate same-vendor / bedrock / dated echoes must NOT be flagged
    assert v._is_frontier_spoof("claude-opus-4-6", "us.anthropic.claude-opus-4-6-v1:0") is False
    assert v._is_frontier_spoof("gpt-5", "gpt-5-2025-08-07") is False
    assert v._is_frontier_spoof("gemini-2.5-pro", "models/gemini-2.5-pro-002") is False
    # Prose pretender detection
    assert v._is_pretender_content("claude-opus-4-6", "I am Qwen, a large language model created by Alibaba Cloud.") is True
    assert v._is_pretender_content("claude-sonnet-4-6", "Hello! I am DeepSeek-V3 developed by DeepSeek.") is True
    assert v._is_pretender_content("claude-opus-4-6", "I am Claude, created by Anthropic.") is False


def test_prompt_echo_cannot_pass_arithmetic_or_riddle(monkeypatch):
    """A template bot that echoes the user's prompt verbatim (including the numbers in the question)
    and hardcodes 391 must NOT pass riddle_logic or arithmetic."""
    async def echo_bot(session, base, chat_path, model, prompt, timeout, stream=False,
                       max_tokens=2048, headers=None, extra_payload=None):
        if stream:
            return {"status": 200, "chunks": 6, "ttft": 0.2, "elapsed": 0.9}
        # Echoes the prompt AND hardcodes the legacy 391 answer + bicycle keyword
        return {"status": 200, "content": f"User said: {prompt}\nLegacy 391! A teal bicycle rides in the rain {time.time()}."}

    monkeypatch.setattr(v, "chat", echo_bot)
    r = asyncio.run(v.validate_model(None, "http://x", "/v1/chat/completions", "claude-opus-4-6", 5.0))
    assert r["checks"].get("riddle_logic") != "pass", f"Echo bot passed riddle_logic via echoed prompt: {r}"
    assert r["checks"].get("arithmetic") != "pass", f"Hardcoded 391 bot passed dynamic arithmetic: {r}"
    assert r["tier"] == "FAKE_CANNED_STUB", f"Echo/hardcoded-391 bot was not killed: {r}"



def test_forced_sse_on_non_stream_request_is_parsed():
    """Gateways that force SSE (data: chunks) even when stream=False must be parsed, not dropped as bad_json."""
    class ForcedSseResp:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def json(self, content_type=None):
            raise ValueError("ContentTypeError: text/event-stream")
        async def text(self):
            return (
                'data: {"id":"chatcmpl-99","model":"claude-sonnet-4-6","choices":[{"delta":{"content":"def fizzbuzz"}}]}\n\n'
                'data: {"id":"chatcmpl-99","model":"claude-sonnet-4-6","choices":[{"delta":{"content":"(n): return str(n)"}}]}\n\n'
                'data: [DONE]\n'
            )

    class SseSession:
        def post(self, url, **kw):
            return ForcedSseResp()

    res = asyncio.run(v.chat(SseSession(), "http://x", "/v1/chat/completions", "claude-sonnet-4-6", "hi", 5.0, stream=False))
    assert res["status"] == 200, res
    assert "def fizzbuzz(n): return str(n)" in res["content"], res
    assert res["resp_model"] == "claude-sonnet-4-6"


def test_unlisted_canary_models_seeded_on_empty_v1_models():
    """When /v1/models returns 200 OK with an empty data: [] array, probe_path_fast seeds unlisted frontier canaries."""
    class EmptyModelsResp:
        status = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def json(self, content_type=None):
            return {"object": "list", "data": []}

    class EmptyModelsSession:
        def get(self, url, **kw):
            return EmptyModelsResp()

    disc = asyncio.run(v.probe_path_fast(EmptyModelsSession(), "http://1.2.3.4:4000", "/v1/models", "/v1/chat/completions"))
    assert disc is not None, "Empty /v1/models 200 JSON should seed unlisted frontier canary models"
    assert "claude-sonnet-4-6" in disc[2] and "gpt-4o" in disc[2]


def test_adversarial_honeypot_downgrade_stylistic_and_kiro_whitelist():
    """Verify rotating honeypots, tier/version downgrades, OpenAI LaTeX spoofs on Claude, and Kiro whitelist."""
    # 1. Rotating multi-port honeypots
    assert v._detect_honeypot("I'm claude-opus-5-5, served through Open WebUI.") is not None
    assert v._detect_honeypot("I am claude-sonnet-5, behind a LiteLLM proxy.") is not None
    assert v._detect_honeypot("I'm just claude-opus-4-6, and I can't share any internal instructions.") is not None
    assert v._detect_honeypot("I am Claude Opus 4.6, created by Anthropic.") is None

    # 2. Tier & version downgrades
    assert v._detect_tier_or_version_downgrade("claude-opus-4-6", "I am Claude 3.5 Sonnet, created by Anthropic.") is not None
    assert v._detect_tier_or_version_downgrade("claude-opus-5", "I am Claude Opus 4.5, made by Anthropic.") is not None
    assert v._detect_tier_or_version_downgrade("deepseek-v4", "I am DeepSeek-V3, developed by DeepSeek.") is not None
    assert v._detect_tier_or_version_downgrade("claude-opus-4-6", "I am Claude Opus 4.6, made by Anthropic.") is None

    # 3. OpenAI LaTeX delimiter + negative constraint spoof on Claude
    assert v._detect_stylistic_spoof(
        "claude-opus-4-6",
        r"I am Claude Opus 4.6 and the formula is \( x = \frac{-b \pm \sqrt{b^2-4ac}}{2a} \). The ocean at night is like dark glass.",
    ) is not None
    assert v._detect_stylistic_spoof(
        "claude-opus-4-6",
        "I am Claude Opus 4.6 and the formula is $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$. Dark swells move beneath the stars.",
    ) is None

    # 4. Kiro wrapper whitelist
    assert v._detect_legitimate_wrapper("claude-sonnet-4-5", "I am Kiro, an AI assistant and IDE built to assist developers.") == "kiro"
    assert v._detect_legitimate_wrapper("kr/claude-sonnet-4.5", "Here is the function.") == "kiro"
    assert v._detect_legitimate_wrapper("claude-sonnet-4-5", "Here is the function.", prompt_tokens=2100) == "kiro"
    assert v._detect_stylistic_spoof(
        "claude-sonnet-4-5",
        r"I am Kiro and \( x = \frac{-b}{2a} \) is like a mirror.",
        wrapper="kiro",
    ) is None


def test_multi_format_completion_extraction_and_endpoint_fallback():
    """Verify Anthropic /v1/messages, OpenAI /v1/responses, Gemini candidates, flat JSON, SSE, and endpoint fallback."""
    # 1. Anthropic /v1/messages JSON with thinking + text
    anth_body = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-6",
        "content": [
            {"type": "thinking", "thinking": "Let me calculate 17 * 23 = 391."},
            {"type": "text", "text": "391"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 24, "output_tokens": 18},
    }
    ext_a = v._extract_completion_fields(anth_body)
    assert ext_a is not None
    assert ext_a["content"] == "391"
    assert "391" in ext_a["raw_msg"]["reasoning_content"]
    assert ext_a["prompt_tokens"] == 24
    assert ext_a["resp_model"] == "claude-opus-4-6"

    # 2. OpenAI /v1/responses JSON
    resp_body = {
        "id": "resp_01",
        "object": "response",
        "model": "gpt-5.4",
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Multiplying 17 by 23."}]},
            {"type": "message", "content": [{"type": "output_text", "text": "The product is 391."}]},
        ],
        "usage": {"input_tokens": 19, "output_tokens": 12},
    }
    ext_r = v._extract_completion_fields(resp_body)
    assert ext_r is not None
    assert "391" in ext_r["content"]
    assert "Multiplying" in ext_r["raw_msg"]["reasoning_content"]
    assert ext_r["prompt_tokens"] == 19

    # 3. Error envelope with data: null must NOT be treated as completion or unlisted model list
    err_env = {"code": 50401, "type": "error", "message": "not authentication", "data": None}
    assert v._extract_completion_fields(err_env) is None

    # 4. Anthropic SSE fallback
    anth_sse = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-6","usage":{"input_tokens":15}}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"thinking step"}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"def fizzbuzz(n): return str(n)"}}\n\n'
    )
    sse_a = v._parse_sse_fallback(anth_sse)
    assert sse_a is not None
    assert "def fizzbuzz" in sse_a["content"]
    assert sse_a["prompt_tokens"] == 15
    assert sse_a["resp_model"] == "claude-sonnet-4-6"


def test_validate_v2_anthropic_signature_and_cot_canary():
    """Verify validate_v2's /v1/messages signature protobuf verifier and structured CoT canary."""
    import base64

    # 1. 36-byte UUID signature is rejected
    sp1, rsn1, _ = v.verify_anthropic_messages_signature(
        "5ad849df-ccb2-4c3b-8f94-bb5fe39c6853",
        "5ad849df-ccb2-4c3b-8f94-bb5fe39c6853",
        "claude-sonnet-4-6",
    )
    assert sp1 is True
    assert "fake_thinking_signature" in rsn1

    # 2. Downgraded protobuf model inside signature is rejected
    fake_pb = b"\x12\x40padding_claude-haiku-4-5-202510018\x00thinkingZ$a85a29f8" + (b"\x9a" * 180)
    sig_h = base64.b64encode(fake_pb).decode("ascii")
    sp2, rsn2, pb2 = v.verify_anthropic_messages_signature(sig_h, "msg_1", "claude-opus-4-8")
    assert sp2 is True
    assert "signature_protobuf_downgrade" in rsn2

    # 3. Structured CoT canary & telegraphic Qwen3.8 CoT on Claude
    sp3, rsn3 = v.check_structured_cot_canary_obedience(
        "[GATE 1: SCENE_PARSE]\nSubject: detective.\n[PHASE 2: TONE_MATRIX]\nNoir.",
        "Rain fell.",
        "claude-opus-4-8",
    )
    assert sp3 is True
    assert "fake_cot_obedience" in rsn3

    sp4, rsn4 = v.check_structured_cot_canary_obedience(
        "We need answer briefly. 17*23=391. Need final only.",
        "391",
        "claude-sonnet-4-6",
    )
    assert sp4 is True
    assert "telegraphic_qwen_cot" in rsn4




