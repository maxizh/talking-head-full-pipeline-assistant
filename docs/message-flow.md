# Chat message flow

How the browser, `agent/server.py`, and `agent/llm.py` talk to each other. The same picture is drawn in [diagrams/agent-message-flow.html](diagrams/agent-message-flow.html).

## Topology

The browser only speaks to the local server:

| Route | Role |
|---|---|
| `GET /` | Chat page (`agent/web/index.html`) |
| `GET /teleprompter` | Prompter page |
| `POST /api/chat` | One user turn, answered as `text/event-stream` |
| `POST /api/cancel` | Ask the in-flight turn to stop between tools |
| `GET /api/events` | Server-sent events for prompter actions (record start/stop) |
| `GET /api/teleprompter/stream` | Script and scroll commands for the prompter |
| `POST /api/record/start`, `POST /api/record/stop` | Prompter record buttons, direct to OBS |
| `POST /api/teleprompter/finished` | Prompter reports that the script scrolled to the end |

`llm.py` is not a service. `run_agent` calls it inside the server process. The server listens on `127.0.0.1` and `PORT` (default `8788`).

## LLM request

`llm.py` POSTs to `{ANTHROPIC_BASE_URL}/v1/messages` with `anthropic-version: 2023-06-01` and `stream: true`.

| Header | When |
|---|---|
| `authorization: Bearer $ANTHROPIC_AUTH_TOKEN` | `ANTHROPIC_AUTH_TOKEN` is set. This wins if both credentials exist. |
| `x-api-key: $ANTHROPIC_API_KEY` | No bearer token, API key is set. |

The body is the Anthropic Messages shape: `model`, `max_tokens`, `messages`, optional `system`, optional `tools`. Streaming events are assembled back into `text` and `tool_use` blocks. A turn retries transient network and HTTP 429/5xx failures only before any text has been forwarded to the browser, so a retry cannot duplicate a half-printed reply.

## One turn

1. The page POSTs `{session, text}`.
2. `run_agent` appends the user message to that session's in-memory history and calls the model with the tool schemas.
3. Text deltas are forwarded as SSE `{type: "text", delta}` while they arrive.
4. If the reply contains `tool_use` blocks, each tool runs locally. The page sees `{type: "tool_call"}` then `{type: "tool_result"}`. Exceptions become a tool result with `is_error: true` and the exception text unchanged.
5. Tool results are appended as a user message and the model is called again.
6. A reply with no tool use ends the turn. The page then receives `{type: "done"}`.
7. After 12 model rounds the server emits `{type: "error"}` and stops.

Session histories live in a process-local dict. One lock serializes turns so a session's message list is not written concurrently. Cancel uses a second lock: the turn lock is held for the whole loop, and a cancel request has to be recorded without waiting for it. Stop takes effect before the next model call or between tools. A tool that already started is not rolled back. If the history ends on an assistant `tool_use` with no result, or on a bare user message, the server patches that history so the next turn is still a valid Messages transcript.

Prompter-side record events are queued and prepended to the next chat message as `[系统事件 …]` lines, so a follow-up like "cut the take I just stopped" can see the file path.

## SSE frames

| `type` | Meaning |
|---|---|
| `text` | Incremental assistant text (`delta`) |
| `tool_call` | Name and arguments about to run |
| `tool_result` | Tool output, or the error string when `ok` is false |
| `cancelled` | The user stopped this turn |
| `error` | The turn failed, or the round limit was hit |
| `done` | The HTTP stream is finished |
