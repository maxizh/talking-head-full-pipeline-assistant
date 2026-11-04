#!/usr/bin/env python3
"""Check that this machine can run Talking-Head Full Pipeline Assistant.

Exit 0 when the required checks pass. Warnings (no transcription engine)
do not fail the run. Fix lines are written for anything that is missing.
"""

import importlib
import os
import shutil
import socket
import sys
import urllib.parse

GREEN, RED, YELLOW, END = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
fails = 0


def check(name, ok, fix="", warn=False):
    global fails
    if ok:
        mark = f"{GREEN}ok{END}"
    elif warn:
        mark = f"{YELLOW}warn{END}"
    else:
        mark = f"{RED}fail{END}"
    print(f"  {mark}  {name}" + ("" if ok else f"\n        {fix}"))
    if not ok and not warn:
        fails += 1


def obs_endpoint():
    url = os.environ.get("OBS_URL", "").strip()
    if url:
        parsed = urllib.parse.urlparse(url)
        return parsed.hostname or "127.0.0.1", parsed.port or 4455
    return "127.0.0.1", int(os.environ.get("OBS_PORT", "4455"))


check(
    f"Python {sys.version.split()[0]} (need 3.12+)",
    sys.version_info >= (3, 12),
    "Install Python 3.12+ and recreate the venv: python3.12 -m venv .venv",
)

for module, package in (
    ("websocket", "websocket-client"),
    ("sklearn", "scikit-learn"),
    ("PIL", "Pillow"),
):
    try:
        importlib.import_module(module)
        check(f"{package}", True)
    except ImportError:
        check(package, False, "pip install -r requirements.txt")

engines = ("mlx_whisper", "faster_whisper", "whisper")
found_engine = None
for name in engines:
    try:
        importlib.import_module(name)
        found_engine = name
        break
    except ImportError:
        continue
check(
    f"transcription engine ({found_engine or 'none'})",
    found_engine is not None,
    "Optional until you transcribe or burn subtitles. "
    "pip install faster-whisper, mlx-whisper, or openai-whisper",
    warn=True,
)

check("ffmpeg", shutil.which("ffmpeg") is not None, "Install ffmpeg and ensure it is on PATH")
check("ffprobe", shutil.which("ffprobe") is not None, "Install ffmpeg; ffprobe ships with it")

host, port = obs_endpoint()
sock = socket.socket()
sock.settimeout(2)
obs_up = sock.connect_ex((host, port)) == 0
sock.close()
check(
    f"OBS WebSocket ({host}:{port})",
    obs_up,
    "Open OBS → Tools → WebSocket Server Settings → enable the server, "
    "then set OBS_PASSWORD in .env (and OBS_URL if it is not ws://localhost:4455).",
)

if obs_up:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from agent.tools import obs

        status = obs.get_status()
        scene = status.get("current_scene") or "?"
        check(f"OBS handshake (OBS {status.get('obs')}, scene {scene!r})", True)
    except Exception as exc:
        check("OBS handshake", False, f"{exc} — check OBS_PASSWORD")

has_cred = bool(os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY"))
check(
    "LLM credential (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN)",
    has_cred,
    "Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in .env, then source env.sh. "
    "PARALLIGHT_API_KEY is accepted by env.sh as an alias when ANTHROPIC_* is empty.",
)

free_gb = shutil.disk_usage(".").free / 1e9
check(
    f"free disk {free_gb:.0f} GB (want ≥ 5)",
    free_gb >= 5,
    "Free some disk space. Recordings and intermediate cuts are large.",
)

print()
http_port = os.environ.get("PORT", "8788")
if fails:
    print(f"{RED}{fails} required check(s) failed.{END}")
    sys.exit(1)
print(f"{GREEN}Ready.{END} Start the assistant: python -m agent.server")
print(f"  chat         http://localhost:{http_port}/")
print(f"  teleprompter http://localhost:{http_port}/teleprompter")
