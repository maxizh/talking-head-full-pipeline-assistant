"""ffmpeg editing tools.

Durations reported to the caller are read back with ffprobe.
Outputs never overwrite a source or an existing file; a name clash gets a numeric suffix.
"""

import json
import os
import re
import subprocess
import tempfile
import time

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

BREATH = 0.15        # 切静音时首尾各留的呼吸，避免吃掉起音
MIN_KEEP = 0.30      # 短于这个的保留片段直接丢掉
COPY_TOLERANCE = 0.25  # 流拷贝切出来的时长与预期差超过这个值就改成重编码精切


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-4:])
        raise RuntimeError(f"{os.path.basename(cmd[0])} 失败（exit {p.returncode}）：{tail}")
    return p


def _require(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return path


def _out_path(src, suffix, out=None):
    if out:
        return os.path.abspath(os.path.expanduser(out))
    stem, ext = os.path.splitext(src)
    return f"{stem}{suffix}{ext}"


def _pick_out(dst, *sources):
    """铁律：永不覆盖、永不写回源文件。撞名就自动加序号，让每一版都留着可回退。"""
    dst = os.path.abspath(dst)
    for s in sources:
        if dst == os.path.abspath(s):
            raise ValueError(f"输出路径就是源文件（{dst}），原片必须保持不动；换一个 out")
    if not os.path.exists(dst):
        return dst
    stem, ext = os.path.splitext(dst)
    n = 2
    while os.path.exists(f"{stem}-{n}{ext}"):
        n += 1
    return f"{stem}-{n}{ext}"


def _record(dst, tool, sources, params):
    """写一份剪辑记录在成片旁边，用来回溯「这条片子是怎么来的」。"""
    with open(dst + ".edit.json", "w") as f:
        json.dump({"file": dst, "produced_by": tool,
                   "sources": [os.path.abspath(s) for s in sources],
                   "params": params, "created": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False)


def edit_history(path, _depth=0):
    """列出某个成片的完整剪辑链路，一路回溯到原始素材。"""
    path = os.path.abspath(os.path.expanduser(path))
    ledger = path + ".edit.json"
    node = {"file": path, "exists": os.path.isfile(path)}
    if not os.path.isfile(ledger):
        node["origin"] = True      # 没有记录 = 原始素材（或外部导入的文件）
        return node
    with open(ledger) as f:
        rec = json.load(f)
    node.update({"produced_by": rec["produced_by"], "params": rec["params"],
                 "created": rec["created"], "origin": False})
    if _depth < 20:
        node["from"] = [edit_history(s, _depth + 1) for s in rec["sources"]]
    return node


SILENT_MAX_DB = -80.0   # 峰值都到不了这个，就是数字静音（麦克风没信号），不是安静


def audio_level(path):
    """读整条音轨的平均 / 峰值音量（dB）。只解码音频，很快。"""
    p = subprocess.run([FFMPEG, "-nostdin", "-v", "info", "-i", path, "-vn",
                        "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    mean = re.search(r"mean_volume:\s*(-?[\d.]+|-inf)", p.stderr)
    peak = re.search(r"max_volume:\s*(-?[\d.]+|-inf)", p.stderr)
    to_db = lambda m: -999.0 if not m or m.group(1) == "-inf" else float(m.group(1))
    mean_db, max_db = to_db(mean), to_db(peak)
    return {"mean_db": mean_db, "max_db": max_db, "silent": max_db < SILENT_MAX_DB}


def probe(path):
    """Read duration, streams, and loudness with ffprobe.

    This is the ground truth for "did the file actually record / export".
    A present audio stream can still be digital silence; `audio.silent` says so.
    """
    path = _require(path)
    p = _run([FFPROBE, "-v", "error", "-show_entries",
              "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate",
              "-of", "json", path])
    info = json.loads(p.stdout)
    streams = info.get("streams", [])
    return {
        "path": path,
        "duration_s": round(float(info["format"]["duration"]), 3),
        "size_bytes": int(info["format"].get("size", 0)),
        "streams": [{k: s.get(k) for k in
                     ("index", "codec_type", "codec_name", "width", "height", "r_frame_rate")
                     if s.get(k) is not None} for s in streams],
        "has_video": any(s.get("codec_type") == "video" for s in streams),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "verified_by_ffprobe": True,
        # 有音轨不等于录到了声音：麦克风没信号时音轨照样存在，只是全程静音
        **({"audio": audio_level(path)}
           if any(s.get("codec_type") == "audio" for s in streams) else {}),
    }


def _duration(path):
    p = _run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nw=1:nk=1", path])
    return round(float(p.stdout.strip()), 3)


def _cut_copy(src, start, end, dst):
    _run([FFMPEG, "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
          "-i", src, "-c", "copy", "-avoid_negative_ts", "make_zero", dst])


def _cut_encode(src, start, end, dst):
    _run([FFMPEG, "-y", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
          "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
          "-c:a", "aac", "-b:a", "160k", dst])


def cut(path, start, end, out=None):
    """切 [start, end)。先用流拷贝（秒级完成），时长对不上再改成重编码精切。"""
    src = _require(path)
    start, end = float(start), float(end)
    if end <= start:
        raise ValueError(f"区间不合法：start={start} 必须小于 end={end}")
    total = _duration(src)
    if start >= total:
        raise ValueError(f"start={start} 超出了视频时长 {total}s")
    end = min(end, total)
    dst = _pick_out(_out_path(src, f"_cut_{start:g}-{end:g}", out), src)

    want = round(end - start, 3)
    _cut_copy(src, start, end, dst)
    got = _duration(dst)
    method = "stream_copy"
    if abs(got - want) > COPY_TOLERANCE:
        # 流拷贝只能在关键帧上下刀，偏差太大就重编码精切
        _cut_encode(src, start, end, dst)
        got = _duration(dst)
        method = "reencode"
    _record(dst, "cut", [src], {"start": start, "end": end, "method": method})
    return {"path": dst, "duration_s": got, "expected_s": want,
            "method": method, "verified_by_ffprobe": True}


def concat(paths, out):
    """把多段按顺序拼起来（流拷贝，失败则重编码）。"""
    srcs = [_require(p) for p in paths]
    if len(srcs) < 2:
        raise ValueError("拼接至少要两段")
    dst = _pick_out(os.path.expanduser(out), *srcs)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for s in srcs:
            f.write("file '%s'\n" % s.replace("'", "'\\''"))
        listfile = f.name
    try:
        cmd = [FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listfile]
        try:
            _run(cmd + ["-c", "copy", dst])
            method = "stream_copy"
        except RuntimeError:
            _run(cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-c:a", "aac", "-b:a", "160k", dst])
            method = "reencode"
    finally:
        os.unlink(listfile)
    _record(dst, "concat", srcs, {"segments": len(srcs), "method": method})
    return {"path": dst, "duration_s": _duration(dst), "segments": len(srcs),
            "method": method, "verified_by_ffprobe": True}


def detect_silences(path, noise_db=-25, min_silence_s=0.8, breath_s=BREATH):
    """找静音（冷场）区间，并反算出该保留的有声区间。首尾各留 breath_s 的呼吸。"""
    src = _require(path)
    info = probe(src)
    if not info["has_audio"]:
        raise RuntimeError(f"{os.path.basename(src)} 没有音轨，没法按静音检测；请改用时间区间直接切。")
    total = info["duration_s"]

    p = subprocess.run(
        [FFMPEG, "-v", "info", "-i", src,
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}", "-f", "null", "-"],
        capture_output=True, text=True)
    if p.returncode != 0:
        tail = "\n".join(p.stderr.strip().splitlines()[-4:])
        raise RuntimeError(f"silencedetect 失败（exit {p.returncode}）：{tail}")

    starts = [float(m) for m in re.findall(r"silence_start:\s*(-?[\d.]+)", p.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*(-?[\d.]+)", p.stderr)]
    silences = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else total  # 静音一直到片尾时没有 silence_end
        silences.append({"start": round(max(0.0, s), 3), "end": round(min(e, total), 3)})

    # 静音区间取反 = 有声区间，再把呼吸加回来
    keep = []
    cursor = 0.0
    for sil in silences:
        if sil["start"] > cursor:
            keep.append([cursor, sil["start"]])
        cursor = max(cursor, sil["end"])
    if cursor < total:
        keep.append([cursor, total])

    spans = []
    for s, e in keep:
        s = max(0.0, s - breath_s)
        e = min(total, e + breath_s)
        if spans and s <= spans[-1][1]:       # 加呼吸后重叠了就合并
            spans[-1][1] = max(spans[-1][1], e)
        elif e - s >= MIN_KEEP:
            spans.append([s, e])
    keep_spans = [{"start": round(s, 3), "end": round(e, 3)} for s, e in spans]
    kept = sum(s["end"] - s["start"] for s in keep_spans)

    return {"path": src, "duration_s": total, "silences": silences,
            "keep_spans": keep_spans, "would_cut_s": round(total - kept, 3),
            "kept_s": round(kept, 3),
            "params": {"noise_db": noise_db, "min_silence_s": min_silence_s,
                       "breath_s": breath_s}}


def cut_spans(path, spans, out=None):
    """按区间列表切出多段再拼成一条；返回的时长是 ffprobe 真读的。"""
    src = _require(path)
    spans = [(float(s["start"]), float(s["end"])) if isinstance(s, dict) else (float(s[0]), float(s[1]))
             for s in spans]
    if not spans:
        raise ValueError("spans 是空的，没有要保留的区间")
    dst = _pick_out(_out_path(src, "_edited", out), src)

    workdir = tempfile.mkdtemp(prefix="edit_")
    try:
        ext = os.path.splitext(src)[1] or ".mp4"
        parts = []
        for i, (s, e) in enumerate(spans):
            seg = os.path.join(workdir, f"part{i:03d}{ext}")
            parts.append(cut(src, s, e, out=seg)["path"])
        if len(parts) == 1:
            _run([FFMPEG, "-y", "-v", "error", "-i", parts[0], "-c", "copy", dst])
            method = "single_span"
        else:
            method = concat(parts, dst)["method"]
    finally:
        for f in os.listdir(workdir):
            os.unlink(os.path.join(workdir, f))
        os.rmdir(workdir)

    # 覆盖掉 concat 写的那份记录：它的 sources 是已删除的临时分段，回溯不到原片
    _record(dst, "cut_spans", [src],
            {"spans": [[round(s, 3), round(e, 3)] for s, e in spans], "method": method})
    got = _duration(dst)
    want = round(sum(e - s for s, e in spans), 3)
    return {"path": dst, "duration_s": got, "expected_s": want,
            "spans": len(spans), "method": method, "verified_by_ffprobe": True}


def _tool(name, fn, description, properties, required):
    return name, {
        "fn": lambda **kw: json.dumps(fn(**kw), ensure_ascii=False),
        "schema": {"name": name, "description": description,
                   "input_schema": {"type": "object", "properties": properties,
                                    "required": required}},
    }


_PATH = {"type": "string", "description": "视频文件绝对路径"}
_SPANS = {
    "type": "array",
    "description": "要保留的区间列表，按时间顺序，如 [{\"start\":0,\"end\":5.2}]",
    "items": {"type": "object",
              "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
              "required": ["start", "end"]},
}

TOOLS = dict([
    _tool("probe", probe,
          "读视频的真实时长、体积、流类型和音量。录完或出片后必须用它验证。"
          "注意 audio.silent=true 表示音轨是纯静音（麦克风没信号），要明确告诉用户，不能当成正常。",
          {"path": _PATH}, ["path"]),
    _tool("cut", cut,
          "切出 [start, end) 一段，返回新文件路径和 ffprobe 真读的时长。优先流拷贝，"
          "时长对不上会自动改成重编码精切。",
          {"path": _PATH, "start": {"type": "number", "description": "起点秒"},
           "end": {"type": "number", "description": "终点秒"},
           "out": {"type": "string", "description": "可选，输出路径"}},
          ["path", "start", "end"]),
    _tool("concat", concat,
          "把多个视频按顺序拼成一条，返回 ffprobe 真读的时长。",
          {"paths": {"type": "array", "items": {"type": "string"},
                     "description": "要拼接的文件路径，按播放顺序"},
           "out": {"type": "string", "description": "输出路径"}},
          ["paths", "out"]),
    _tool("detect_silences", detect_silences,
          "检测静音（冷场）区间，并反算出该保留的有声区间 keep_spans（首尾各留 0.15 秒呼吸，"
          "避免吃掉起音）。返回 silences、keep_spans、would_cut_s。不改文件，只分析。",
          {"path": _PATH,
           "noise_db": {"type": "number", "description":
                        "低于多少 dB 算静音，默认 -25（适合有环境底噪的真人录音）。"
                        "数值越大判定越宽松、检出的冷场越多，-20 往往过头会剪掉正常说话；"
                        "数值越小越严格，-30 只适合干净录音，-40 检出更少。"
                        "检出 0 段时应调大，不是调小。"},
           "min_silence_s": {"type": "number", "description": "至少持续多久才算冷场，默认 0.8 秒"},
           "breath_s": {"type": "number", "description": "首尾呼吸秒数，默认 0.15"}},
          ["path"]),
    _tool("cut_spans", cut_spans,
          "按区间列表切出多段再拼成一条成片（有副作用：写新文件）。通常接 detect_silences 的 "
          "keep_spans。返回 ffprobe 真读的时长。",
          {"path": _PATH, "spans": _SPANS,
           "out": {"type": "string", "description": "可选，输出路径"}},
          ["path", "spans"]),
    _tool("edit_history", edit_history,
          "列出某个成片的完整剪辑链路，一路回溯到原始素材：每一步是哪个工具、什么参数、什么时候做的。"
          "剪错了用它找回上一版（原片和中间版本都不会被覆盖，一直留在磁盘上）。",
          {"path": _PATH}, ["path"]),
])
