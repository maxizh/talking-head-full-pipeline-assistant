[中文](README.md) | English

# Talking-Head Full Pipeline Assistant

A talking-head assistant that runs on this machine: write a spoken script from a topic, record it in OBS while you follow a teleprompter, then automatically cut dead air and mistakes and burn in subtitles.

The chat page, the tool loop, the corpus, and the finished videos stay on this machine. Only two things leave it: the model you configure (the Anthropic Messages API, or a compatible gateway), and web pages you explicitly ask it to fetch.

## Features

- **Scripts.** Articles are fetched into a local corpus, retrieved with character-level TF-IDF (mixed Chinese and English), then written as a spoken script. A paragraph can end with `[来源:文件名]` (source: filename). Facts that are not in the retrieval results should not be invented.
- **Teleprompter.** The script is pushed to the fullscreen page `/teleprompter`, which can scroll, pause, and return to the top. On macOS, starting a recording brings this page to the front.
- **OBS recording.** Scenes, inputs, filters, start, and stop are controlled through obs-websocket 5.x. Device ids and kinds follow OBS's own enumeration. After recording stops, ffprobe checks the duration, picture, and audio track, and warns when the track is digital silence.
- **Auto edit.** Cut by silence, or cut by transcript segment number. The model reports only segment numbers; the tool looks up the timestamps. If too much would be deleted, a long consecutive stretch would be removed, or key points of the script would disappear from the finished video, it first issues a confirmation token and writes the file only after you agree. The finished video is always a new file. `edit_history` can trace back to the original.
- **Subtitles.** Pillow draws a rounded semi-transparent bar, and ffmpeg composites it onto the picture. Timestamps come from a transcription of the current file, so what gets burned in is the edited video, not the raw footage. macOS prefers PingFang, Hiragino Sans GB (冬青黑体), or STHeiti (华文黑体). Linux falls back to WenQuanYi Micro Hei (文泉驿微米黑), Noto Sans CJK, or Droid Sans Fallback.
- **Chat window.** A single Chinese page. SSE streams the reply text and every tool call, and you can stop in the middle. Steps that have already finished are not undone.

Local integrations for recording (launching OBS, bringing the teleprompter to the front, `caffeinate` to prevent sleep, and sleep/wake notifications) are implemented for **macOS**. The chat service, editing, transcription, and retrieval themselves run as long as Python 3.12 and ffmpeg are available.

## Directory structure

```
.
  LICENSE                       # MIT
  README.md
  requirements.txt              # runtime dependencies
  .env.example                  # environment template; copy to .env
  .gitignore
  env.sh                        # loads .env and maps legacy variable names
  preflight.py                  # environment check
  .claude/
    launch.json                 # local launch: assistant server, preflight
  agent/
    __init__.py
    server.py                   # chat window + agent loop
    llm.py                      # Anthropic Messages client
    tools/
      __init__.py
      obs.py                    # raw obs-websocket protocol
      edit.py                   # ffmpeg editing + silence detection
      transcribe.py             # transcription + cut by segment
      research.py               # corpus + retrieval
      subtitle.py               # subtitle cards, burned in with ffmpeg
    web/
      index.html                # chat page
      teleprompter.html         # teleprompter
  docs/
    message-flow.md             # chat requests and SSE
    obs-control.md              # OBS protocol and tools
    diagrams/
      agent-message-flow.html   # message-flow diagram
```

After you run it, these paths also appear and are not committed: `.venv/`, `.env`, `corpus/` (the corpus), recordings, and intermediate edit files.

How a request goes from the browser to the model and the tools: [docs/message-flow.md](docs/message-flow.md). The OBS client: [docs/obs-control.md](docs/obs-control.md).

## Requirements

- Python 3.12 or newer
- [ffmpeg](https://ffmpeg.org/) and `ffprobe` on `PATH`
- [OBS Studio](https://obsproject.com/) 28 or newer, with the built-in WebSocket server enabled (Tools → WebSocket Server Settings)
- Model credentials: `ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN` for a gateway
- A transcription engine (needed only for cutting by segment and burning subtitles). Install one of:
  - Apple Silicon: `mlx-whisper`
  - General: `faster-whisper`
  - or `openai-whisper`

## Environment setup

The project is fixed on Python 3.12. If the system `python3` is older, do not install dependencies with it. Create a separate virtual environment.

```bash
cd <project-directory>
uv venv --python 3.12 .venv
# Without uv: python3.12 -m venv .venv

.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install faster-whisper
# On Apple Silicon, you can install this instead: .venv/bin/python -m pip install mlx-whisper

cp .env.example .env
```

Edit `.env`. The program reads only `ANTHROPIC_*` and the OBS / runtime variables below.

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | one of the two | Direct Anthropic connection; request header `x-api-key` |
| `ANTHROPIC_AUTH_TOKEN` | one of the two | Compatible gateway; request header `Authorization: Bearer`. When both are set, this one is used |
| `ANTHROPIC_BASE_URL` | no | Default `https://api.anthropic.com`. Do not append a path |
| `ANTHROPIC_MODEL` | no | Default `claude-sonnet-4-20250514` |
| `OBS_PASSWORD` | required if OBS authentication is on | WebSocket password |
| `OBS_URL` | no | Default `ws://localhost:4455` |
| `OBS_PORT` | no | Used by `preflight.py` when `OBS_URL` is unset. Default `4455` |
| `PORT` | no | Local server port. Default `8788` |
| `CORPUS_DIR` | no | Corpus directory. Default `./corpus` |
| `TRANSCRIBE_MODEL` | no | Whisper model spec. Default `small` |
| `FFMPEG` / `FFPROBE` | no | Executable name or absolute path |

An older environment that sets only `PARALLIGHT_BASE_URL` and `PARALLIGHT_API_KEY` still works. `source env.sh` maps them to `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` only when the corresponding `ANTHROPIC_*` value is empty. `ANTHROPIC_*` values that are already set are left as they are. A new `.env` should use `ANTHROPIC_*` directly.

**In every new terminal, run this first:**

```bash
cd <project-directory> && source env.sh && .venv/bin/python <script>
```

- `source env.sh` reads `.env`, applies the `PARALLIGHT_*` legacy names as described above, and adds Homebrew ffmpeg (Apple Silicon `/opt/homebrew/bin`, Intel `/usr/local/bin`) plus `.venv/bin` to `PATH`. Environment variables do not carry into a new terminal, so source it every time. The script does not print secrets.
- `.venv/bin/python` always uses Python 3.12 from the project virtual environment, so the system `python3` is left alone.

**Check that the environment is ready:**

```bash
cd <project-directory> && source env.sh && .venv/bin/python preflight.py
```

When every check passes, it prints the chat page and teleprompter addresses. If OBS is not running, or the credentials are not filled in yet, the matching check fails. Follow the message, then run it again. A missing transcription engine is only a warning. Install one before editing and burning subtitles.

## Install and run

After the environment is set up:

```bash
cd <project-directory> && source env.sh && .venv/bin/python preflight.py
cd <project-directory> && source env.sh && .venv/bin/python -m agent.server
```

- Chat page: <http://localhost:8788/>
- Teleprompter: <http://localhost:8788/teleprompter>

The server listens only on `127.0.0.1`. To change the port:

```bash
cd <project-directory> && source env.sh && PORT=8790 .venv/bin/python -m agent.server
```

## Usage

In the chat page, say these in order.

1. Give a topic and article URLs. The assistant fetches the article text (`fetch_article`), searches (`corpus_search`), and writes a short spoken script with sources.
2. Push the script to the teleprompter (`teleprompter_load`). If the page is not open, open `/teleprompter` yourself. Do not use full-screen capture, or the teleprompter will appear in the picture. Window capture or application capture is fine, as long as the captured window is not the teleprompter window.
3. Ask it to start recording. It checks OBS, then starts. On macOS, if the teleprompter already has a script, it brings that page to the front. The button on the chat page or on the teleprompter can stop the recording and returns `outputPath`. If the microphone has no signal, it says so immediately.
4. Ask it to tighten the finished video. Dead air is `detect_silences`, then `cut_spans`. Cutting by content is `transcribe`, then `cut_segments`: the deleted original text is listed item by item. When a cut is blocked, the reply includes a `confirm_token`, and the file is written only after you agree.
5. Burn subtitles onto the **edited** video (`burn_subtitles`). To check type size and line wrapping first, use `preview_subtitle_card`.

To judge whether this corpus is worth a vector store, have it run `retrieval_experiment`. The `lsa` in the report is a dimensionality reduction of this corpus's TF-IDF, not a trained vector model.

## License

[MIT](LICENSE).
