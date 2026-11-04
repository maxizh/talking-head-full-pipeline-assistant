"""Anthropic Messages API client.

Stdlib urllib, streaming SSE. Reads ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN
(Bearer, preferred), ANTHROPIC_API_KEY (x-api-key), and ANTHROPIC_MODEL.
Legacy PARALLIGHT_* names are mapped by env.sh before this process starts;
this module does not read them.
"""

import http.client
import json
import os
import ssl
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
DEFAULT_MAX_TOKENS = 2048


def _endpoint():
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    return base + "/v1/messages"


def _headers():
    h = {
        "content-type": "application/json",
        "accept": "text/event-stream",
        "anthropic-version": "2023-06-01",
    }
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if token:
        h["authorization"] = "Bearer " + token
    elif key:
        h["x-api-key"] = key
    else:
        raise RuntimeError(
            "没有凭据：在 .env 设置 ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN，然后 source env.sh"
        )
    return h


RETRIES = 3
RETRY_HTTP = {429, 500, 502, 503, 504, 529}
# 网络层的偶发故障：DNS 解析失败、TLS 握手被掐断、连接被重置、读到一半断开
TRANSIENT = (urllib.error.URLError, ConnectionError, TimeoutError, ssl.SSLError,
             http.client.HTTPException)


class _RetryableHTTP(Exception):
    pass


def call(messages, tools=None, system=None, model=None, max_tokens=DEFAULT_MAX_TOKENS,
         on_text=None, should_stop=None):
    """调一轮模型，返回装配好的 assistant 消息；网络偶发故障自动重试。

    on_text(delta) 在每个文字增量到达时被调用，用来往前端推 text 帧。
    should_stop() 每收一帧调一次，用来在吐字过程中中途停下（抛异常即可终止）。
    返回 {"content": [block, ...], "stop_reason": str}，content 里是 text / tool_use 块。

    只在「还没往前端吐出任何字」时重试：已经吐了一半再重来，前端会看到重复的文字。
    """
    emitted = False

    def relay(delta):
        nonlocal emitted
        emitted = True
        if on_text:
            on_text(delta)

    for attempt in range(1, RETRIES + 1):
        try:
            return _call_once(messages, tools, system, model, max_tokens, relay, should_stop)
        except (_RetryableHTTP, *TRANSIENT) as e:
            if emitted or attempt == RETRIES:
                raise RuntimeError(
                    f"连不上大模型网关（已试 {attempt} 次）：{e}。"
                    "多半是网络或 DNS 抖动，稍后再发一次。") from None
            time.sleep(1.5 * attempt)


def _call_once(messages, tools, system, model, max_tokens, on_text, should_stop):
    body = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    if system:
        body["system"] = system

    req = urllib.request.Request(
        _endpoint(), data=json.dumps(body).encode(), headers=_headers(), method="POST"
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        if e.code in RETRY_HTTP:
            raise _RetryableHTTP(f"HTTP {e.code}: {detail[:120]}") from None
        raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from None

    blocks = []
    partial_json = {}
    stop_reason = None

    with resp:
        for raw in resp:
            if should_stop:
                should_stop()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            ev = json.loads(payload)
            kind = ev.get("type")

            if kind == "content_block_start":
                blk = dict(ev["content_block"])
                idx = ev["index"]
                while len(blocks) <= idx:
                    blocks.append(None)
                if blk["type"] == "text":
                    blk.setdefault("text", "")
                elif blk["type"] == "tool_use":
                    partial_json[idx] = ""
                blocks[idx] = blk

            elif kind == "content_block_delta":
                idx = ev["index"]
                delta = ev["delta"]
                if delta["type"] == "text_delta":
                    blocks[idx]["text"] += delta["text"]
                    if on_text:
                        on_text(delta["text"])
                elif delta["type"] == "input_json_delta":
                    partial_json[idx] += delta["partial_json"]

            elif kind == "content_block_stop":
                idx = ev["index"]
                if blocks[idx] and blocks[idx]["type"] == "tool_use":
                    raw_args = partial_json.get(idx, "").strip()
                    blocks[idx]["input"] = json.loads(raw_args) if raw_args else {}

            elif kind == "message_delta":
                stop_reason = ev.get("delta", {}).get("stop_reason", stop_reason)

            elif kind == "error":
                raise RuntimeError(f"LLM 返回错误: {ev.get('error')}")

    return {"content": [b for b in blocks if b], "stop_reason": stop_reason}
