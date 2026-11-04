# OBS control

`agent/tools/obs.py` speaks [obs-websocket 5.x](https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md) with `websocket-client`. There is no third-party OBS SDK. `agent/server.py` registers one tool per request type and wraps `StartRecord` / `StopRecord` so the prompter, the sleep assertion, and the silence check stay in step with the recording.

```
chat  →  server tool loop  →  obs.request()  →  ws://localhost:4455
                                              subprotocol obswebsocket.json
```

`preflight.py` calls `obs.get_status()` and expects `obs` (version) and `current_scene`.

## Connection

Each request opens a connection and closes it when the response arrives. A single call is sub-millisecond once OBS is up, and the server's turn lock already allows only one OBS call at a time, so a long-lived socket is not required. The URL is `OBS_URL` (default `ws://localhost:4455`).

On macOS, `launch_if_needed()` starts `OBS.app` in the background (`open -g -j -a OBS`) when the port is closed and the URL points at localhost. It does not click through the WebSocket settings for you. Enable the server in OBS and put the password in `OBS_PASSWORD`.

## Handshake

1. Connect with subprotocol `obswebsocket.json`. Without it, some OBS builds answer with a non-JSON frame.
2. Read op `0` Hello. If `authentication` is present, require `OBS_PASSWORD` and send op `1` Identify:
   - `secret = base64(sha256(password + salt))`
   - `authentication = base64(sha256(secret + challenge))`
3. Wait for op `2` Identified. A wrong password makes OBS close the socket with no frame. That close is reported as a password error, not as a generic disconnect.

## Requests

Send op `6` with `requestType`, a `requestId`, and `requestData`. Keep reading until op `7` carries the same `requestId`. Op `5` events on the same socket are ignored. `requestStatus.result == false` raises, and the tool loop returns that text to the model as an error result.

`StopRecord` can return `outputPath` before OBS has finished flushing the file. Callers that probe the file wait and retry when the probe does not yet match a finished recording. The system prompt tells the model to do the same.

## Tools

Recording: `GetVersion`, `GetRecordStatus`, `GetSceneList`, `GetCurrentProgramScene`, `SetCurrentProgramScene`, `StartRecord`, `StopRecord`.

Scene setup: `GetInputList`, `GetInputKindList`, `CreateInput`, `GetInputPropertiesListPropertyItems`, `GetInputSettings`, `SetInputSettings`, `GetSourceFilterKindList`, `CreateSourceFilter`, `SetSourceFilterSettings`.

The model is instructed to read lists before it writes. Camera device ids come from `GetInputPropertiesListPropertyItems` (`propertyName=device` for the macOS `av_capture_input` family; the exact kind string is whatever `GetInputKindList` returns on that OBS build). Brightness uses filter kind `color_filter_v2`.

`screen_capture` covers three modes. `GetInputSettings` field `type` is `0` full display, `1` window, `2` application. Kind name alone does not mean the teleprompter is in frame. Querying `propertyName=application` on a `screen_capture` input can stall OBS for many seconds; use `window` to pick a window.

## Recording side effects

`StartRecord` from chat also marks the server recording state, asks macOS `caffeinate` to hold idle sleep for at most four hours (no-op when `caffeinate` is absent), and focuses the teleprompter when a script is loaded.

`StopRecord` clears that state and, two seconds later, runs `volumedetect`. A peak under -80 dB is treated as a dead microphone and pushed to the chat and the prompter.

Lid-close and manual sleep are not blocked. A watchdog thread treats a wall-clock jump longer than 30 seconds as a wake, then asks OBS whether recording is still active and posts a notification when a take was in progress.
