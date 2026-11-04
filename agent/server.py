"""HTTP server and tool loop for Talking-Head Full Pipeline Assistant (口播全流程助手).

Stdlib `http.server` only. From the repository root:

    source env.sh && python -m agent.server

Chat: http://localhost:8788/  (override with PORT). The process binds to 127.0.0.1.
"""

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import llm
from agent.tools import edit, obs, research, subtitle, transcribe

WEB_DIR = Path(__file__).parent / "web"
PORT = int(os.environ.get("PORT", 8788))
MAX_ROUNDS = 12
TP_HEARTBEAT_S = 15
SYSTEM_PROMPT = (
    "你是口播全流程助手（Talking-Head Full Pipeline Assistant），跑在用户本机。"
    "你负责口播稿、OBS 录制、提词器、剪辑和字幕。\n"
    "纪律：\n"
    "1. 有副作用的操作（开录、停录、建源、加滤镜、改设置）先复述你要做什么。\n"
    "2. 工具报错原样透出，不要粉饰或猜测结果。\n"
    "3. 布场景每一步都用列表工具核实（GetInputKindList、GetInputPropertiesListPropertyItems、"
    "GetInputList、GetSourceFilterKindList、GetSceneList），绝不凭空猜设备 id、kind 名或场景名。\n"
    "4. 枚举不到设备（列表为空或只有空项）就明确告诉用户没找到，并提示检查设备连接和 OBS 的摄像头权限；"
    "不要声称在用默认设备，也不要把没绑定设备的源报告成完成。\n"
    "5. 停录后把 outputPath 告诉用户。\n"
    "6. 录制或出片完成后，必须用 probe 真读文件来验证（时长、有没有画面和音轨），"
    "不要只信工具返回值；停录后文件可能还在写，probe 结果不对就等一下再读一次。\n"
    "7. 按内容选剪时，你绝不自己产时间戳，只报 transcribe 给出的段编号，"
    "编号到时间戳的换算由工具查表完成。\n"
    "8. 讲错后重讲的，删前面讲错的、留最后一次重讲；口头禅和跑题段可以删。\n"
    "9. 只要是按内容删，就把 removed_segments 里删掉的原文逐条贴给用户看——不管删多删少。"
    "人扫一眼就知道那句是废话还是核心论点，这比任何阈值都准。\n"
    "10. cut_segments 拒绝出片时（删太多 / 连续大块删除 / 有口播稿要点丢失），先把清单和 "
    "block_reasons 给用户看，等用户明确同意后才原样回传 confirm_token 调第二次，"
    "不要自己替用户确认。\n"
    "11. 剪辑永远写新文件，原片和中间版本一律不动、不覆盖（撞名自动加序号）。"
    "剪错了用 edit_history 查这条片子的剪辑链路，回到上一版重剪。\n"
    "12. 「主题 → 口播稿 → 提词器」的流程：先 fetch_article 抓资料存进语料库，"
    "再 corpus_search 检索，然后写稿，最后 teleprompter_load 推给提词器页面。"
    "推过去的稿子会成为后续剪辑的锚点：删到稿子要点的段落会被拦下来。\n"
    "13. 写口播稿：口语化短句，用自己的话组织，不整段照抄原文；每段末尾标 [来源:文件名]；"
    "检索结果里没有的事实一律不写，宁可少说也不编。\n"
    "14. 用户要求比较检索方式，或问这份语料该不该上向量库时："
    "先 corpus_list / corpus_search 看语料里实际有什么，"
    "再设计 4~6 组问题（每组一个用原文措辞的问法、一个换同义词的问法，"
    "外加一小段必然出现在正确答案里的原文用于判命中），调 retrieval_experiment，"
    "把 keyword / tfidf / lsa / llm_tfidf 整理成并排表格。"
    "结论必须写明 lsa 只是对本语料做的局部降维，不是训练过的向量模型；"
    "只有换说法问法上 tfidf 明显掉点、而 lsa 明显更好时，才建议上真正的向量检索。\n"
    "15. 烧字幕永远对剪好的成片做，不要对原始素材做——字幕时间戳来自 burn_subtitles "
    "对传入文件自己的重新转录，拿原片烧出来的字幕会和剪过的画面整体错位。\n"
    "16. 提词器不能入镜：开录前用 GetInputList 查场景里的源，遇到屏幕采集类的源"
    "（screen_capture / display_capture）必须再用 GetInputSettings 看它的 type —— "
    "0 是整屏采集，提词器一定会入镜，要警告用户改用窗口采集或双屏；"
    "1（窗口）和 2（应用）只采集指定目标，不必拦，但要提醒用户别把提词器开在被采集的那个窗口里。"
    "光看 kind 名叫 screen_capture 就报警是误报。"
)

SESSIONS = {}
LOCK = threading.Lock()

# 用户中途喊停：LOCK 在整轮 agent 循环期间一直被占着，
# 所以取消标记必须用另一把锁，否则「停止」请求会卡在等锁上，永远停不下来。
CANCELLING = set()
CANCEL_LOCK = threading.Lock()


class Cancelled(Exception):
    """用户在任务跑到一半时按了停止。"""


def cancel_session(session_id):
    with CANCEL_LOCK:
        CANCELLING.add(session_id)


def _clear_cancel(session_id):
    with CANCEL_LOCK:
        CANCELLING.discard(session_id)


def _repair_history_after_cancel(history):
    """停在半路会留下残缺历史，补齐后这个会话才能继续用。

    两种残缺：模型已发出 tool_use 但结果没回填（下次调用会直接报错）；
    以及最后一条是用户消息、没有助手回复（角色不交替）。
    """
    if not history:
        return
    last = history[-1]
    if last["role"] == "assistant" and isinstance(last["content"], list):
        pending = [b for b in last["content"]
                   if isinstance(b, dict) and b.get("type") == "tool_use"]
        if pending:
            history.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": b["id"],
                 "content": "用户按了停止，这个工具没有执行", "is_error": True}
                for b in pending]})
    if history[-1]["role"] == "user":
        history.append({"role": "assistant",
                        "content": [{"type": "text", "text": "（这一步被用户停止了）"}]})


def _stop_if_cancelled(session_id):
    with CANCEL_LOCK:
        hit = session_id in CANCELLING
    if hit:
        raise Cancelled()

# 提词器：当前稿子 + 所有打开着的页面（每个订阅者一个队列）
# 稿子落盘，免得重启服务时把准备好的稿子弄丢（改代码重启是常事）
TP_SCRIPT_FILE = Path(__file__).parent / ".teleprompter_script.txt"
TELEPROMPTER = {"script": "", "subscribers": [], "finished_at": 0.0, "pending": None}
TP_PENDING_S = 30   # 没页面在线时发的控制指令，30 秒内有页面连上就补发
TP_LOCK = threading.Lock()

if TP_SCRIPT_FILE.exists():
    TELEPROMPTER["script"] = TP_SCRIPT_FILE.read_text()
    research.set_script(TELEPROMPTER["script"])


# 系统事件：提词器页面上的操作（比如按了停录）要让对话窗口也知道。
# subscribers 是开着的对话页；pending 攒给 agent，下一轮对话时作为上下文带进去，
# 这样用户接着说「剪一下刚才那条」时，agent 知道那条是哪个文件。
EVENTS = {"subscribers": [], "pending": []}
EV_LOCK = threading.Lock()


def broadcast_event(event):
    event = {**event, "at": time.strftime("%H:%M:%S")}
    with EV_LOCK:
        subs = list(EVENTS["subscribers"])
        EVENTS["pending"].append(event)
        del EVENTS["pending"][:-20]
    for q in subs:
        q.put(event)


def _drain_events():
    with EV_LOCK:
        pending, EVENTS["pending"] = EVENTS["pending"], []
    return pending


def _tp_broadcast(frame):
    with TP_LOCK:
        subs = list(TELEPROMPTER["subscribers"])
    for q in subs:
        q.put(frame)
    return len(subs)


def teleprompter_load(script):
    """把口播稿推给提词器页面，同时记为剪辑锚点。"""
    with TP_LOCK:
        TELEPROMPTER["script"] = script
        TELEPROMPTER["finished_at"] = 0.0   # 新稿子还没念，别沿用上一份稿子「已到底」的状态
    TP_SCRIPT_FILE.write_text(script)
    points = research.set_script(script)   # 剪到这些要点时 cut_segments 会拦下来
    viewers = _tp_broadcast({"type": "script", "text": script})
    return {"chars": len(script), "script_points": points, "viewers": viewers,
            "url": f"http://localhost:{PORT}/teleprompter",
            "hint": "没人看到就让用户打开这个地址" if not viewers else "已推送给打开着的提词器页面"}


# 录制状态缓存：每次调用 OBS 都是一次连接建立+断开，OBS 会弹提示。
# 页面刷新只是想知道「在不在录」，没必要为此敲一次 OBS。
_REC = {"active": None, "at": 0.0}

# 录制期间不让电脑睡眠：睡一下录像就断一截。
# caffeinate 挡得住空闲睡眠，挡不住合盖和手动「睡眠」——那两种靠唤醒后的检查兜底。
# -w 跟着本进程走（服务退出它就退出），-t 封顶 4 小时，防止 OBS 在别处停录后一直不让睡。
_caffeinate = None
CAFFEINATE_MAX_S = 4 * 3600


def _check_recording_audio(path):
    """停录后查一下录到的声音。麦克风没信号（发射端没开、没电、没配对）时，
    录像照样有音轨，只是全程静音——不提醒的话，要到剪辑时才发现整条白录了。"""
    time.sleep(2)   # 停录后 OBS 还要一会儿才把文件写完
    try:
        level = edit.audio_level(path)
    except Exception:
        return
    if not level["silent"]:
        return
    msg = (f"刚录的这条没有声音（峰值 {level['max_db']:.0f} dB，麦克风没有信号）。"
           "检查麦克风有没有开机、有没有电、有没有配对，或者在 OBS 里换一个音频输入设备。")
    _notify("录音是无声的", msg)
    broadcast_event({"type": "record_silent", "text": msg, "outputPath": path})
    _tp_broadcast({"type": "record", "ok": False, "error": msg})


def _set_recording(active, output_path=None):
    """所有开录 / 停录入口都走这里：记录状态、开关防睡眠、同步提词器页面的按钮。

    页面同步必须放在这里：从对话里开录时如果不通知页面，页面上的「停录」一直是灰的，点不动。
    """
    global _caffeinate
    _REC.update(active=active, at=time.time())
    frame = {"type": "record", "ok": True, "active": active}
    if output_path:
        frame["outputPath"] = output_path
    _tp_broadcast(frame)
    if not active and output_path:
        threading.Thread(target=_check_recording_audio, args=(output_path,),
                         daemon=True).start()
    if active and (_caffeinate is None or _caffeinate.poll() is not None):
        # macOS only. Missing caffeinate must not fail a recording that OBS already started.
        if shutil.which("caffeinate"):
            _caffeinate = subprocess.Popen(
                ["caffeinate", "-dims", "-t", str(CAFFEINATE_MAX_S), "-w", str(os.getpid())])
    elif not active and _caffeinate is not None:
        _caffeinate.terminate()
        _caffeinate = None
REC_CACHE_S = 10.0


def _record_state(max_age=REC_CACHE_S):
    if _REC["active"] is not None and time.time() - _REC["at"] < max_age:
        return {"type": "record", "ok": True, "active": _REC["active"], "cached": True}
    try:
        active = bool(obs.request("GetRecordStatus").get("outputActive"))
        _REC.update(active=active, at=time.time())
        return {"type": "record", "ok": True, "active": active}
    except Exception as e:
        return {"type": "record", "ok": False, "error": f"OBS 连不上：{e}"}


# 先在已打开的浏览器里找到提词器那个标签页并激活它。
# 只用 `open` 不够：它把 URL 交给「默认浏览器」，页面开在别的浏览器里时
# 会另开一个新页面，而用户正在看的窗口根本不动。
# 浏览器名必须写死在脚本里：AppleScript 用变量装应用名就丢了术语支持，
# tabs / active tab index 这些词直接编译不过（syntax error -2740）。
_BROWSERS = ("Google Chrome", "Google Chrome for Testing", "Chromium",
             "Microsoft Edge", "Brave Browser", "Safari")
_installed_browsers = None


def _osascript(args, timeout=4, **kw):
    """osascript 必须带超时：没给自动化权限时它会一直卡在系统授权弹窗上，
    之前就是这样把开录整个挂死的（录制已开始，工具却永远不返回）。"""
    try:
        return subprocess.run(["osascript", *args], capture_output=True, text=True,
                              timeout=timeout, **kw)
    except subprocess.TimeoutExpired:
        return None


def _installed():
    """只保留本机装了的浏览器：脚本是整体编译的，引用没装的应用会让整段编译失败，try 也救不了。"""
    global _installed_browsers
    if _installed_browsers is None:
        _installed_browsers = []
        for n in _BROWSERS:
            r = _osascript(["-e", f'id of application "{n}"'], timeout=3)
            if r is not None and r.returncode == 0:
                _installed_browsers.append(n)
    return _installed_browsers


def _tab_script(url, action):
    """生成找到提词器标签后执行 action 的脚本：focus 切到前台，reload 刷新。"""
    chromium = {"focus": "set active tab index of w to i\n            set index of w to 1\n"
                         "            activate",
                "reload": "tell t to reload"}[action]
    safari = {"focus": "set current tab of w to t\n            set index of w to 1\n"
                       "            activate",
              "reload": "set URL of t to (URL of t)"}[action]
    blocks = [f'''try
  if application "{name}" is running then
    tell application "{name}"
      repeat with w in windows
        set i to 0
        repeat with t in tabs of w
          set i to i + 1
          if URL of t starts with "{url}" then
            {chromium}
            return "done:{name}"
          end if
        end repeat
      end repeat
    end tell
  end if
end try''' for name in _installed() if name != "Safari"]
    if "Safari" in _installed():
        blocks.append(f'''try
  if application "Safari" is running then
    tell application "Safari"
      repeat with w in windows
        repeat with t in tabs of w
          if URL of t starts with "{url}" then
            {safari}
            return "done:Safari"
          end if
        end repeat
      end repeat
    end tell
  end if
end try''')
    return "\n".join(blocks) + '\nreturn "notfound"\n'


def _wait_viewers(timeout):
    """等提词器页面连上服务，返回在线页面数（超时返回 0）。"""
    end = time.time() + timeout
    while True:
        with TP_LOCK:
            n = len(TELEPROMPTER["subscribers"])
        if n or time.time() >= end:
            return n
        time.sleep(0.25)


def focus_teleprompter():
    """把提词器页面切到前台；已经开着就激活那个标签，没开才新开一个。"""
    url = f"http://localhost:{PORT}/teleprompter"
    with TP_LOCK:
        viewers = len(TELEPROMPTER["subscribers"])

    found = _osascript(["-"], input=_tab_script(url, "focus"), timeout=5)
    verdict = found.stdout.strip() if found else "timeout"
    if verdict.startswith("done:"):
        browser = verdict.split(":", 1)[1]
        # 切到前台不等于连着：后台标签被浏览器冻结过、或者服务重启过，
        # 页面里的推送连接可能已经断了，这时发「开始滚动」没人收得到
        n = _wait_viewers(3)
        if n:
            return {"url": url, "viewers_before": viewers, "viewers_after": n,
                    "how": f"激活了已打开的标签（{browser}）"}
        _osascript(["-"], input=_tab_script(url, "reload"), timeout=5)
        n = _wait_viewers(6)
        if n:
            return {"url": url, "viewers_before": viewers, "viewers_after": n,
                    "how": f"激活并刷新了已打开的标签（{browser}，原来的连接已经断了）"}
        verdict = "stale"

    # 没找到已打开的标签、标签救不回来、或没有自动化权限：交给默认浏览器新开
    p = subprocess.run(["open", url], capture_output=True, text=True, timeout=10)
    if p.returncode != 0:
        raise RuntimeError(f"打开提词器失败：{p.stderr.strip() or p.returncode}")
    n = _wait_viewers(6)
    note = {
        "timeout": ("（没能激活已打开的标签：osascript 超时，多半是 macOS 自动化权限没给 —— "
                    "系统设置 → 隐私与安全性 → 自动化，允许控制浏览器）"),
        "notfound": "",
        "stale": "（已打开的标签刷新后还是连不上服务）",
    }.get(verdict, f"（找标签失败：{(found.stderr or '').strip()[:80]}）")
    return {"url": url, "viewers_before": viewers, "viewers_after": n,
            "how": f"在默认浏览器里新开了页面{note}"}


def teleprompter_control(action, speed=None):
    """控制提词器：start 开始滚动 / pause 暂停 / reset 回到顶部。"""
    if action not in ("start", "pause", "reset"):
        raise ValueError(f"action 只能是 start / pause / reset，收到 {action!r}")
    if action in ("start", "reset"):
        # 重新开滚就清掉上一轮的「已到底」，否则等「滚完再停录」的地方会以为一开始就滚完了
        with TP_LOCK:
            TELEPROMPTER["finished_at"] = 0.0
    frame = {"type": "control", "action": action}
    if speed is not None:
        frame["speed"] = float(speed)
    viewers = _tp_broadcast(frame)
    if viewers:
        return {"action": action, "speed": speed, "viewers": viewers}
    # 没页面在线：存起来，页面一连上就补发，别让「开始滚动」凭空消失
    with TP_LOCK:
        TELEPROMPTER["pending"] = (time.time(), frame)
    return {"action": action, "speed": speed, "viewers": 0,
            "note": f"现在没有提词器页面在线，指令已暂存，{TP_PENDING_S} 秒内页面连上会自动补发"}


def echo(text):
    """Return text unchanged. Use it to confirm the tool loop, not as a product step."""
    return text


TOOLS = {
    "echo": {
        "fn": echo,
        "schema": {
            "name": "echo",
            "description": "回声测试工具：把传入的 text 原样返回，用于验证工具调用链路。",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "要回显的文字"}},
                "required": ["text"],
            },
        },
    }
}
TOOLS.update(obs.TOOLS)
TOOLS.update(edit.TOOLS)
TOOLS.update(transcribe.TOOLS)
TOOLS.update(research.TOOLS)
TOOLS.update(subtitle.TOOLS)

# 有口播稿在提词器上时，开录就自动把提词器切到前台 —— 做成开录动作的一部分，
# 不指望模型每次都记得多调一步。
_obs_start_record = TOOLS["StartRecord"]["fn"]


def _start_record_then_focus(**kw):
    result = json.loads(_obs_start_record(**kw))
    _set_recording(True)
    with TP_LOCK:
        has_script = bool(TELEPROMPTER["script"])
    if has_script:
        try:
            result["teleprompter"] = focus_teleprompter()
        except Exception as e:
            result["teleprompter_error"] = f"{type(e).__name__}: {e}"
    return json.dumps(result, ensure_ascii=False)


TOOLS["StartRecord"]["fn"] = _start_record_then_focus
TOOLS["StartRecord"]["schema"]["description"] += (
    " 提词器上有口播稿时，开录会自动把提词器页面切到前台（没打开的话顺手打开）。"
    " 录制期间电脑不会自动睡眠。")

_obs_stop_record = TOOLS["StopRecord"]["fn"]


def _stop_record_and_release(**kw):
    result = _obs_stop_record(**kw)
    _set_recording(False, json.loads(result).get("outputPath"))   # 停录了才放开防睡眠
    return result


TOOLS["StopRecord"]["fn"] = _stop_record_and_release
TOOLS.update({
    "teleprompter_load": {
        "fn": lambda **kw: json.dumps(teleprompter_load(**kw), ensure_ascii=False),
        "schema": {
            "name": "teleprompter_load",
            "description": "把口播稿推送到提词器页面（/teleprompter），所有打开着的页面立即更新。"
                           "段落之间用空行分隔，每段末尾标 [来源:文件名]。",
            "input_schema": {"type": "object",
                             "properties": {"script": {"type": "string",
                                                       "description": "完整口播稿"}},
                             "required": ["script"]},
        },
    },
    "focus_teleprompter": {
        "fn": lambda **kw: json.dumps(focus_teleprompter(**kw), ensure_ascii=False),
        "schema": {
            "name": "focus_teleprompter",
            "description": "把提词器页面切到前台；没打开就打开它。开录时会自动调用，"
                           "用户说「看不到提词器」时也可以手动调。",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    },
    "teleprompter_control": {
        "fn": lambda **kw: json.dumps(teleprompter_control(**kw), ensure_ascii=False),
        "schema": {
            "name": "teleprompter_control",
            "description": "控制提词器滚动：start 开始 / pause 暂停 / reset 回顶，可同时设速度。",
            "input_schema": {
                "type": "object",
                "properties": {"action": {"type": "string",
                                          "enum": ["start", "pause", "reset"]},
                               "speed": {"type": "number",
                                         "description": "滚动速度 px/s，常用 40~120"}},
                "required": ["action"]},
        },
    },
})


def run_agent(session_id, text, emit):
    """一轮对话的 agent 循环：调模型 → 有 tool_use 就执行并回填 → 再调，直到纯文本。"""
    _clear_cancel(session_id)
    with LOCK:
        history = SESSIONS.setdefault(session_id, [])
        history.append({"role": "user", "content": text})

        for _ in range(MAX_ROUNDS):
            _stop_if_cancelled(session_id)
            reply = llm.call(
                history,
                tools=[t["schema"] for t in TOOLS.values()],
                system=SYSTEM_PROMPT,
                on_text=lambda delta: emit("text", {"delta": delta}),
                should_stop=lambda: _stop_if_cancelled(session_id),
            )
            history.append({"role": "assistant", "content": reply["content"]})

            tool_uses = [b for b in reply["content"] if b["type"] == "tool_use"]
            if not tool_uses:
                return

            results = []
            for call_block in tool_uses:
                # 停止只在工具之间生效：已经发出去的副作用（比如已经开录）不会被撤销
                _stop_if_cancelled(session_id)
                name, args = call_block["name"], call_block["input"]
                emit("tool_call", {"id": call_block["id"], "name": name, "args": args})
                try:
                    if name not in TOOLS:
                        raise KeyError(f"未注册的工具: {name}")
                    output = str(TOOLS[name]["fn"](**args))
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": call_block["id"],
                        "content": output,
                    })
                    emit("tool_result", {"id": call_block["id"], "name": name,
                                         "ok": True, "content": output})
                except Exception as e:
                    message = f"{type(e).__name__}: {e}"
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": call_block["id"],
                        "content": message,
                        "is_error": True,
                    })
                    emit("tool_result", {"id": call_block["id"], "name": name,
                                         "ok": False, "content": message})

            history.append({"role": "user", "content": results})

        emit("error", {"message": f"连续 {MAX_ROUNDS} 轮都在调工具，已中止"})


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        route = self.path.split("?")[0]
        pages = {"/": "index.html", "/teleprompter": "teleprompter.html"}
        if route in pages:
            page = (WEB_DIR / pages[route]).read_bytes()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
        elif route == "/api/teleprompter/stream":
            self.stream_teleprompter()
        elif route == "/api/events":
            self.stream_events()
        elif route == "/api/teleprompter/state":
            with TP_LOCK:
                self.send_json({"viewers": len(TELEPROMPTER["subscribers"]),
                                "script_chars": len(TELEPROMPTER["script"]),
                                "finished_at": TELEPROMPTER["finished_at"]})
        else:
            self.send_error(404)

    def stream_events(self):
        """对话窗口订阅系统事件：提词器页面上的操作在这边也能实时看到。"""
        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        q = queue.Queue()
        with EV_LOCK:
            EVENTS["subscribers"].append(q)
        try:
            while True:
                try:
                    payload = (b"data: " + json.dumps(q.get(timeout=TP_HEARTBEAT_S),
                                                      ensure_ascii=False).encode() + b"\n\n")
                except queue.Empty:
                    payload = b": ping\n\n"
                self.wfile.write(payload)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with EV_LOCK:
                if q in EVENTS["subscribers"]:
                    EVENTS["subscribers"].remove(q)

    def stream_teleprompter(self):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        q = queue.Queue()
        with TP_LOCK:
            TELEPROMPTER["subscribers"].append(q)
            current = TELEPROMPTER["script"]
        try:
            # 新开的页面先补上当前稿子和录制状态，不用等下一次推送；
            # 再补上没人在线时暂存的控制指令（比如开录后紧接着的「开始滚动」）
            frames = [{"type": "script", "text": current}, _record_state()]
            with TP_LOCK:
                pending, TELEPROMPTER["pending"] = TELEPROMPTER["pending"], None
            if pending and time.time() - pending[0] < TP_PENDING_S:
                frames.append(pending[1])
            for frame in frames:
                self.wfile.write(b"data: " + json.dumps(
                    frame, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()
            while True:
                try:
                    frame = q.get(timeout=TP_HEARTBEAT_S)
                    payload = b"data: " + json.dumps(frame, ensure_ascii=False).encode() + b"\n\n"
                except queue.Empty:
                    payload = b": ping\n\n"   # 心跳，防止连接被中间层掐掉
                self.wfile.write(payload)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with TP_LOCK:
                if q in TELEPROMPTER["subscribers"]:
                    TELEPROMPTER["subscribers"].remove(q)

    def send_json(self, payload, code=200):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_record(self, stop):
        """提词器页面上的开录 / 停录按钮。"""
        try:
            if stop:
                data = obs.request("StopRecord")
                payload = {"ok": True, "active": False,
                           "outputPath": data.get("outputPath")}
            else:
                obs.request("StartRecord")
                payload = {"ok": True, "active": True}
            _set_recording(payload["active"], payload.get("outputPath"))
        except Exception as e:
            payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            _REC.update(active=None, at=0.0)   # 状态不明，下次别用缓存
            _tp_broadcast({"type": "record", **payload})   # 失败也让页面显示原因
        self.send_json(payload)
        if payload.get("ok"):
            broadcast_event({
                "type": "record",
                "text": (f"提词器页面：已停录，文件 {payload.get('outputPath')}"
                         if payload.get("active") is False else "提词器页面：已开录"),
                "outputPath": payload.get("outputPath"),
            })

    def do_POST(self):
        route = self.path.split("?")[0]
        if route in ("/api/record/start", "/api/record/stop"):
            self.handle_record(route.endswith("stop"))
            return
        if route == "/api/teleprompter/finished":
            with TP_LOCK:
                TELEPROMPTER["finished_at"] = time.time()
            _tp_broadcast({"type": "finished"})
            self.send_json({"ok": True})
            return
        if route == "/api/cancel":
            length = int(self.headers.get("content-length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            cancel_session(payload.get("session") or "default")
            self.send_json({"ok": True})
            return
        if route != "/api/chat":
            self.send_error(404)
            return

        length = int(self.headers.get("content-length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        session_id = payload.get("session") or "default"
        text = (payload.get("text") or "").strip()

        self.send_response(200)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        def emit(frame_type, data=None):
            frame = {"type": frame_type, **(data or {})}
            self.wfile.write(b"data: " + json.dumps(frame, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        try:
            if not text:
                raise ValueError("空消息")
            # 把这段时间里提词器页面上发生的事带给 agent，
            # 否则用户接着说「剪一下刚才那条」时它根本不知道有过这回事
            events = _drain_events()
            if events:
                note = "\n".join(f"[系统事件 {e['at']}] {e['text']}" for e in events)
                text = f"{note}\n\n{text}"
            run_agent(session_id, text, emit)
        except Cancelled:
            with LOCK:
                _repair_history_after_cancel(SESSIONS.get(session_id, []))
            emit("cancelled", {"message": "已停止这一步。之前完成的步骤不受影响，"
                                          "这一步里已经做掉的操作不会自动撤销。"})
        except Exception as e:
            emit("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            _clear_cancel(session_id)
            emit("done")

    def log_message(self, fmt, *args):
        print(f"[server] {fmt % args}", flush=True)   # 不 flush 的话日志会被缓冲住，排查时看不到


# ---- 睡眠唤醒检查 ----
# 合盖、手动「睡眠」挡不住，只能在醒来后告诉用户录像断了。
# 判断方式：线程每 5 秒醒一次，墙钟一下子跳过去很多，说明中间整机睡过。
SLEEP_GAP_S = 30


def _notify(title, text):
    """macOS 系统通知（右上角弹窗 + 提示音）。"""
    script = (f"display notification {json.dumps(text, ensure_ascii=False)} "
              f"with title {json.dumps(title, ensure_ascii=False)} sound name \"Basso\"")
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _check_after_wake(slept_s):
    was = _REC["active"]
    now = None
    for _ in range(3):   # 刚醒时网络和 OBS 可能还没缓过来
        try:
            now = bool(obs.request("GetRecordStatus").get("outputActive"))
            break
        except Exception:
            time.sleep(5)
    if not (was or now):
        return None
    mins = max(1, round(slept_s / 60))
    if now is False:
        msg = f"电脑睡眠了约 {mins} 分钟，录制已经中断，睡眠之后的内容没有录上。"
    elif now:
        msg = (f"电脑睡眠了约 {mins} 分钟。录制还在继续，"
               "但睡眠期间没有录到内容，片子中间会断一段。")
    else:
        msg = f"电脑睡眠了约 {mins} 分钟，现在连不上 OBS，录制可能已经中断。"
    _notify("录制被睡眠打断", msg)
    broadcast_event({"type": "record_interrupted", "text": msg})
    if now is not None:
        _set_recording(now)
    return msg


def _sleep_watchdog():
    last = time.time()
    while True:
        time.sleep(5)
        gap = time.time() - last
        if gap > SLEEP_GAP_S:
            _check_after_wake(gap)
        last = time.time()


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[server] 聊天窗口: http://localhost:{PORT}/  (模型 {llm.DEFAULT_MODEL})")
    print(f"[server] 提词器:   http://localhost:{PORT}/teleprompter")
    # OBS 冷启动要好几秒，放后台线程，别让对话窗口等它
    threading.Thread(
        target=lambda: print(f"[server] OBS: {obs.launch_if_needed()}", flush=True),
        daemon=True).start()
    threading.Thread(target=_sleep_watchdog, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
