"""obs-websocket 5.x client.

One connection per request, closed when the response arrives.
Protocol notes: docs/obs-control.md.
"""

import base64
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.parse
import uuid

import websocket

OBS_URL = os.environ.get("OBS_URL", "ws://localhost:4455")

OP_HELLO, OP_IDENTIFY, OP_IDENTIFIED = 0, 1, 2
OP_EVENT, OP_REQUEST, OP_RESPONSE = 5, 6, 7


def _b64_sha256(text):
    return base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()


def _recv_json(ws):
    raw = ws.recv()
    if not raw:
        raise websocket.WebSocketConnectionClosedException("empty frame")
    return json.loads(raw)


def _connect(timeout):
    try:
        ws = websocket.create_connection(
            OBS_URL, subprotocols=["obswebsocket.json"], timeout=timeout
        )
    except (ConnectionRefusedError, OSError) as e:
        raise RuntimeError(
            f"连不上 OBS（{OBS_URL}）：OBS 没开，或没在 Tools → WebSocket Server Settings 里启用。({e})"
        ) from None

    try:
        hello = _recv_json(ws)
        if hello.get("op") != OP_HELLO:
            raise RuntimeError(f"OBS 握手异常：第一帧不是 Hello，而是 {hello}")

        identify = {"rpcVersion": 1, "eventSubscriptions": 0}
        auth = hello["d"].get("authentication")
        if auth:
            password = os.environ.get("OBS_PASSWORD")
            if not password:
                raise RuntimeError("OBS 开启了认证，但没设置 OBS_PASSWORD（写入 .env 后 source env.sh）")
            secret = _b64_sha256(password + auth["salt"])
            identify["authentication"] = _b64_sha256(secret + auth["challenge"])

        ws.send(json.dumps({"op": OP_IDENTIFY, "d": identify}))
        try:
            reply = _recv_json(ws)
        except (websocket.WebSocketConnectionClosedException, ConnectionResetError):
            raise RuntimeError("OBS 密码错了：认证失败后 OBS 直接断开了连接，检查 OBS_PASSWORD") from None
        if reply.get("op") != OP_IDENTIFIED:
            raise RuntimeError(f"OBS 握手异常：期望 Identified，收到 {reply}")
        return ws
    except Exception:
        ws.close()
        raise


def _ws_port_open():
    u = urllib.parse.urlparse(OBS_URL)
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((u.hostname or "localhost", u.port or 4455)) == 0


def launch_if_needed(wait_s=40):
    """OBS 没开就在后台隐藏启动（不抢焦点、不弹窗），并等 websocket 端口就绪。"""
    if _ws_port_open():
        return "OBS 已在运行"
    host = urllib.parse.urlparse(OBS_URL).hostname
    if host not in ("localhost", "127.0.0.1", None):
        return f"OBS 配置在别的主机（{host}），不在本机自动启动"
    r = subprocess.run(["open", "-g", "-j", "-a", "OBS"], capture_output=True, text=True)
    if r.returncode != 0:
        return f"启动 OBS 失败：{r.stderr.strip() or '找不到 OBS.app'}"
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if _ws_port_open():
            return "已在后台启动 OBS，websocket 已就绪"
        time.sleep(1)
    return (f"已启动 OBS，但 {wait_s} 秒内 websocket 端口没起来——"
            "检查 OBS 的 Tools → WebSocket Server Settings 是否启用")


def request(request_type, data=None, timeout=10):
    """发一个 OBS 请求，返回 responseData；OBS 报失败时抛 RuntimeError。"""
    ws = _connect(timeout)
    try:
        request_id = uuid.uuid4().hex
        ws.send(json.dumps({
            "op": OP_REQUEST,
            "d": {"requestType": request_type, "requestId": request_id,
                  "requestData": data or {}},
        }))
        while True:
            frame = _recv_json(ws)
            if frame.get("op") != OP_RESPONSE or frame["d"].get("requestId") != request_id:
                continue
            d = frame["d"]
            status = d.get("requestStatus", {})
            if not status.get("result"):
                raise RuntimeError(
                    f"OBS {request_type} 失败 (code {status.get('code')}): {status.get('comment', '无说明')}"
                )
            return d.get("responseData") or {}
    finally:
        ws.close()


def get_status():
    version = request("GetVersion")
    scene = request("GetCurrentProgramScene")
    record = request("GetRecordStatus")
    return {
        "obs": version.get("obsVersion"),
        "websocket": version.get("obsWebSocketVersion"),
        "current_scene": scene.get("currentProgramSceneName") or scene.get("sceneName"),
        "recording": record.get("outputActive", False),
    }


def _tool(request_type, description, properties=None, required=None):
    def fn(**kwargs):
        return json.dumps(request(request_type, kwargs), ensure_ascii=False)

    return request_type, {
        "fn": fn,
        "schema": {
            "name": request_type,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


_STR = {"type": "string"}
_OBJ = {"type": "object"}

TOOLS = dict([
    _tool("GetVersion", "查询 OBS 与 obs-websocket 的版本，以及支持的请求列表。"),
    _tool("GetRecordStatus", "查询录制状态：是否在录、已录时长、字节数。"),
    _tool("GetSceneList", "列出所有场景及当前节目场景。"),
    _tool("GetCurrentProgramScene", "查询当前正在输出（节目）的场景名。"),
    _tool("SetCurrentProgramScene", "切换节目场景。场景名必须来自 GetSceneList。",
          {"sceneName": _STR}, ["sceneName"]),
    _tool("StartRecord", "开始录制（有副作用）。开录前先用 GetRecordStatus 确认没在录。"),
    _tool("StopRecord", "停止录制（有副作用），返回 outputPath：录好的文件路径。"),
    _tool("GetInputList", "列出所有输入源（摄像头、屏幕、麦克风等），可按 inputKind 过滤。",
          {"inputKind": {"type": "string", "description": "可选，只列该 kind 的源"}}),
    _tool("GetInputKindList", "列出本机 OBS 支持的输入源 kind 名。macOS 摄像头是 av_capture_input 系列"
          "（OBS 32 上叫 av_capture_input_v2），以本工具返回的实际名称为准。",
          {"unversioned": {"type": "boolean", "description": "true 时返回不带版本后缀的 kind 名"}}),
    _tool("CreateInput",
          "在场景里新建输入源（有副作用）。inputKind 必须来自 GetInputKindList；"
          "摄像头设备 id 必须先用 GetInputPropertiesListPropertyItems 查到。",
          {"sceneName": _STR, "inputName": _STR, "inputKind": _STR,
           "inputSettings": {**_OBJ, "description": "如摄像头 {\"device\": \"<真实设备 id>\"}"},
           "sceneItemEnabled": {"type": "boolean"}},
          ["sceneName", "inputName", "inputKind"]),
    _tool("GetInputPropertiesListPropertyItems",
          "枚举某个输入源某个下拉属性的可选值。查摄像头设备：propertyName=device（macOS av_capture_input 系列）。"
          "返回为空或只有空项，说明 OBS 看不到任何设备（多半是没给 OBS 摄像头/屏幕录制权限）。"
          "注意：对 screen_capture 源查 propertyName=application 会让 OBS 卡住十几秒，别查它；"
          "要选窗口用 propertyName=window。",
          {"inputName": _STR, "propertyName": _STR}, ["inputName", "propertyName"]),
    _tool("GetInputSettings", "读一个输入源的当前设置。判断屏幕采集到底是哪种模式全靠它："
          "screen_capture 这个 kind 同时涵盖三种模式，settings.type 0=整屏、1=窗口、2=应用，"
          "只看 kind 名会把窗口采集误判成整屏。",
          {"inputName": _STR}, ["inputName"]),
    _tool("SetInputSettings", "修改输入源设置（有副作用），如切换摄像头 device。",
          {"inputName": _STR, "inputSettings": _OBJ,
           "overlay": {"type": "boolean", "description": "true=合并到现有设置（默认），false=整体替换"}},
          ["inputName", "inputSettings"]),
    _tool("GetSourceFilterKindList", "列出本机 OBS 支持的滤镜 kind 名。调亮度用 color_filter_v2。"),
    _tool("CreateSourceFilter",
          "给源加滤镜（有副作用）。filterKind 必须来自 GetSourceFilterKindList；"
          "调亮：filterKind=color_filter_v2, filterSettings={\"brightness\": 0.15}。",
          {"sourceName": _STR, "filterName": _STR, "filterKind": _STR, "filterSettings": _OBJ},
          ["sourceName", "filterName", "filterKind"]),
    _tool("SetSourceFilterSettings", "修改已有滤镜的参数（有副作用），如调整 brightness。",
          {"sourceName": _STR, "filterName": _STR, "filterSettings": _OBJ,
           "overlay": {"type": "boolean"}},
          ["sourceName", "filterName", "filterSettings"]),
])
