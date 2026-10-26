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
