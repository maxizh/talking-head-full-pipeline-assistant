"""Transcription and segment-id editing.

The model names segment ids only. This module looks up timestamps.
It never trusts a model-invented timecode.
"""

import hashlib
import importlib
import json
import os
import re
import subprocess
import tempfile

from agent.tools import edit, research

BRIDGE_S = 0.5      # 相邻保留段之间缝隙小于这个值就桥接，碎切比卡顿更难听
DEFAULT_MODEL = os.environ.get("TRANSCRIBE_MODEL", "small")

# 依次探测，谁装了用谁
ENGINES = ("mlx_whisper", "faster_whisper", "whisper")
_MLX_REPOS = {"tiny": "mlx-community/whisper-tiny-mlx",
              "base": "mlx-community/whisper-base-mlx",
              "small": "mlx-community/whisper-small-mlx",
              "medium": "mlx-community/whisper-medium-mlx",
              "large": "mlx-community/whisper-large-v3-mlx"}

_cache = {}         # (path, mtime, model) -> 转录结果


def _pick_engine():
    for name in ENGINES:
        try:
            importlib.import_module(name)
            return name
        except ImportError:
            continue
    raise RuntimeError(
        "没有可用的转录引擎。装一个即可（按推荐顺序）："
        "pip install mlx-whisper（Apple Silicon 最快）/ faster-whisper / openai-whisper"
    )


def _extract_audio(src):
    """抽 16k 单声道 wav —— 所有引擎都吃这个格式。"""
    fd, wav = tempfile.mkstemp(suffix=".wav", prefix="asr_")
    os.close(fd)
    subprocess.run(
        [edit.FFMPEG, "-y", "-v", "error", "-i", src,
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
        capture_output=True, text=True, check=True)
    return wav


PROMPT_CHARS = 600   # whisper 的提示上限约 224 token，取稿子开头这么多就够


def _vocabulary_prompt():
    """把当前口播稿当成词汇提示喂给 whisper。

    Without a hint, proper nouns are routinely misheard (FFmpeg as "F-Fupeg",
    ffprobe as "F-Probe") and those errors get burned into captions.
    The script already spells them correctly.
    """
    text = research.get_script_text()
    text = re.sub(r"\[来源[:：][^\]]*\]", "", text)
    return " ".join(text.split())[:PROMPT_CHARS] or None


def _run_engine(engine, wav, model, language, prompt=None):
    if engine == "mlx_whisper":
        import mlx_whisper
        repo = _MLX_REPOS.get(model, model)
        r = mlx_whisper.transcribe(wav, path_or_hf_repo=repo, word_timestamps=True,
                                   language=language, initial_prompt=prompt)
        return r["segments"], r.get("language", language)

    if engine == "faster_whisper":
        from faster_whisper import WhisperModel
        segs, info = WhisperModel(model, device="cpu", compute_type="int8").transcribe(
            wav, word_timestamps=True, language=language, initial_prompt=prompt)
        out = [{"start": s.start, "end": s.end, "text": s.text,
                "words": [{"word": w.word, "start": w.start, "end": w.end}
                          for w in (s.words or [])]} for s in segs]
        return out, info.language

    import whisper
    r = whisper.load_model(model).transcribe(wav, word_timestamps=True, language=language,
                                             initial_prompt=prompt)
    return r["segments"], r.get("language", language)


def transcribe(path, language=None, model=None, vocabulary_hint=False):
    """转录，返回带编号和时间戳的段落列表。词级时间戳另存文件，不进对话上下文。

    vocabulary_hint is for subtitle burn-in only. Do not enable it on the
    transcript used for cutting: a clean script prompt makes Whisper swallow
    filler and merge a mistaken take with the correction (the start of the
    retake gets glued onto the bad take), so a segment cut drops good audio.
    """
    src = edit._require(path)
    model = model or DEFAULT_MODEL
    prompt = _vocabulary_prompt() if vocabulary_hint else None
    key = (src, os.path.getmtime(src), model, prompt)   # 换了稿子要重新转录
    if key in _cache:
        return _cache[key]

    engine = _pick_engine()
    wav = _extract_audio(src)
    try:
        raw, language = _run_engine(engine, wav, model, language, prompt)
    finally:
        os.unlink(wav)

    segments, words = [], []
    for i, s in enumerate(raw):
        segments.append({"id": i, "start": round(float(s["start"]), 3),
                         "end": round(float(s["end"]), 3), "text": s["text"].strip()})
        words.append({"id": i, "words": [
            {"word": w["word"], "start": round(float(w["start"]), 3),
             "end": round(float(w["end"]), 3)}
            for w in (s.get("words") or []) if w.get("start") is not None]})

    words_file = src + ".words.json"
    with open(words_file, "w") as f:
        json.dump(words, f, ensure_ascii=False)

    result = {"path": src, "engine": engine, "model": model, "language": language,
              "duration_s": edit._duration(src), "segment_count": len(segments),
              "segments": segments, "words_file": words_file}
    _cache[key] = result
    return result


def _lookup(path, ids):
    """段编号 → 时间戳查表。编号不存在直接报错，绝不猜。"""
    table = {s["id"]: s for s in transcribe(path)["segments"]}
    missing = [i for i in ids if i not in table]
    if missing:
        raise ValueError(
            f"段编号不存在: {missing}；有效编号是 0~{max(table)}（共 {len(table)} 段）")
    return [table[i] for i in sorted(set(ids))]


def segment_spans(path, ids):
    """只查表不出片：给定段编号，返回它们的时间戳（桥接前）。"""
    segs = _lookup(path, ids)
    return {"path": edit._require(path),
            "spans": [{"id": s["id"], "start": s["start"], "end": s["end"],
                       "text": s["text"]} for s in segs]}


def _bridge(segs, bridge_s, dropped=()):
    """相邻保留段之间缝隙 < bridge_s 就连起来，避免碎切。

    缝隙里夹着被删的段就不桥 —— 否则那段会被重新带回成片，
    而清单却声称它删掉了。
    """
    spans = []
    for s in segs:
        gap_start = spans[-1]["end"] if spans else None
        blocked = gap_start is not None and any(
            d["start"] < s["start"] and d["end"] > gap_start for d in dropped)
        if spans and not blocked and s["start"] - spans[-1]["end"] < bridge_s:
            spans[-1]["end"] = max(spans[-1]["end"], s["end"])
            spans[-1]["ids"].append(s["id"])
        else:
            spans.append({"start": s["start"], "end": s["end"], "ids": [s["id"]]})
    return spans


CONFIRM_RATIO = 0.40   # 删掉的内容超过这个比例要拦
CONTIG_SEGMENTS = 3    # 连续删掉这么多段就是整段砍掉，不看比例直接拦
CONTIG_SECONDS = 15.0  # 或者连续删掉这么长时间


def _plan_token(src, kept_ids, spans, dst):
    """把确认绑到具体方案上：换了文件、换了保留段、换了输出路径，token 就失效。"""
    raw = json.dumps([os.path.abspath(src), sorted(kept_ids), dst,
                      [[round(s["start"], 3), round(s["end"], 3)] for s in spans]],
                     sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _blocks(removed, kept_segs, all_count):
    """算出哪些情况必须先给人看。不做语义判断，全靠可数的信号 + 稿子锚点。"""
    reasons = []
    ratio = len(removed) / all_count
    if ratio > CONFIRM_RATIO:
        reasons.append({"type": "ratio",
                        "detail": f"删掉 {ratio:.0%} 的内容（{len(removed)}/{all_count} 段）"})

    # 连续大块删除：不管占比多少，整段砍掉都该看一眼
    run = []
    for s in removed:
        if run and s["id"] == run[-1]["id"] + 1:
            run.append(s)
        else:
            run = [s]
        span_s = run[-1]["end"] - run[0]["start"]
        if len(run) >= CONTIG_SEGMENTS or span_s >= CONTIG_SECONDS:
            reasons.append({
                "type": "contiguous",
                "detail": f"连续删掉第 {run[0]['id']}–{run[-1]['id']} 段"
                          f"（{len(run)} 段 / {span_s:.1f} 秒）"})
            run = []

    # 稿子锚点：有要点在成片里彻底丢了（讲错重讲的那次不算，因为重讲还留着）
    for lost in research.lost_script_points([s["text"] for s in kept_segs],
                                            [s["text"] for s in removed]):
        seg = removed[lost["removed_index"]]
        reasons.append({"type": "lost_script_point", "segment_id": seg["id"],
                        "detail": f"删掉第 {seg['id']} 段会让这个口播稿要点在成片里消失"
                                  f"（保留部分对它的覆盖度只有 {lost['kept_score']}）：{lost['point']}"})
    return reasons


def cut_segments(path, keep_ids=None, drop_ids=None, out=None, bridge_s=BRIDGE_S,
                 dry_run=False, confirm_token=None):
    """按段编号出片（有副作用）。keep_ids 和 drop_ids 二选一。返回 ffprobe 真读时长。

    每次都返回删掉的原文清单。命中拦截条件（占比超 40% / 连续大块删除 /
    删到口播稿要点）时只返回清单和 confirm_token，要原样回传 token 才会出片。
    """
    if (keep_ids is None) == (drop_ids is None):
        raise ValueError("keep_ids 和 drop_ids 必须且只能给一个")
    src = edit._require(path)
    segments = transcribe(src)["segments"]
    all_ids = [s["id"] for s in segments]
    if drop_ids is not None:
        _lookup(src, drop_ids)           # 先校验编号存在
        keep_ids = [i for i in all_ids if i not in set(drop_ids)]
    if not keep_ids:
        raise ValueError("一段都没保留，出不了片")

    segs = _lookup(src, keep_ids)
    kept = {s["id"] for s in segs}
    removed = [s for s in segments if s["id"] not in kept]
    spans = _bridge(segs, bridge_s, dropped=removed)
    total = edit._duration(src)
    dst = edit._out_path(src, "_speechcut", out)
    token = _plan_token(src, kept, spans, dst)
    plan = {
        "path": src,
        "output_path": dst,
        "source_duration_s": total,
        "kept_ids": sorted(kept),
        "removed_ids": [s["id"] for s in removed],
        "removed_segment_ratio": round(len(removed) / len(all_ids), 3),
        "planned_duration_s": round(sum(s["end"] - s["start"] for s in spans), 3),
        "spans": spans,
        "removed_segments": [{"id": s["id"], "text": s["text"]} for s in removed],
    }

    plan["block_reasons"] = _blocks(removed, segs, len(all_ids))

    if dry_run:
        return {**plan, "dry_run": True, "confirm_token": token}
    if plan["block_reasons"] and confirm_token != token:
        wrong = " 你回传的 token 和这个方案不匹配（方案变了，旧 token 作废）。" if confirm_token else ""
        return {**plan, "cut": False, "needs_confirmation": True, "confirm_token": token,
                "message": "没有出片，因为：" +
                           "；".join(r["detail"] for r in plan["block_reasons"]) +
                           f"。{wrong} 先把 removed_segments 里的原文逐条贴给用户看，"
                           f"用户同意后带 confirm_token='{token}' 再调一次。"}

    result = edit.cut_spans(src, [{"start": s["start"], "end": s["end"]} for s in spans],
                            out=dst)
    result.update({k: v for k, v in plan.items() if k != "path"})
    result["removed_s"] = round(total - result["duration_s"], 3)
    # 时长比例含被顺带去掉的段间静音，判断"删太多"要看内容比例
    result["removed_duration_ratio"] = round((total - result["duration_s"]) / total, 3)
    result["must_show_user"] = "把 removed_segments 里删掉的原文逐条给用户看一眼"
    return result


def _tool(name, fn, description, properties, required):
    return name, {
        "fn": lambda **kw: json.dumps(fn(**kw), ensure_ascii=False),
        "schema": {"name": name, "description": description,
                   "input_schema": {"type": "object", "properties": properties,
                                    "required": required}},
    }


_PATH = {"type": "string", "description": "视频/音频文件绝对路径"}
_IDS = {"type": "array", "items": {"type": "integer"},
        "description": "transcribe 返回的段编号（整数），不是时间戳"}

TOOLS = dict([
    _tool("transcribe", transcribe,
          "转录出带编号和时间戳的段落列表 [{id,start,end,text}]。后续所有选剪都基于这里的 id，"
          "你不需要也不可以自己计算时间戳。词级时间戳另存文件，不返回。结果会缓存，重复调用不会重跑。",
          {"path": _PATH,
           "language": {"type": "string", "description":
                        "可选，如 zh / en，默认自动识别。不确定就别传——"
                        "不要照用户提问用的语言来猜音频语言，指定错了转录质量会变差"},
           "model": {"type": "string", "description": "可选，模型规格，默认 small"}},
          ["path"]),
    _tool("segment_spans", segment_spans,
          "查表：给定段编号，返回它们对应的时间戳和原文。只查不改，用来向用户复述你要删/留哪些内容。",
          {"path": _PATH, "ids": _IDS}, ["path", "ids"]),
    _tool("cut_segments", cut_segments,
          "按段编号出片（有副作用）。keep_ids（要保留的）和 drop_ids（要删掉的）二选一给一个；"
          "相邻保留段之间小于 0.5 秒的缝隙会自动桥接，避免碎切。"
          "返回 ffprobe 真读的时长、removed_segment_ratio（删掉的内容比例，判断删太多看这个）"
          "和 removed_duration_ratio（时长缩短比例，含顺带去掉的段间静音，通常更大）。"
          "每次都会在 removed_segments 里返回删掉的原文——无论删多删少，都要逐条贴给用户看。"
          "命中任一条件就不出片，只返回 block_reasons、清单和 confirm_token："
          "删掉内容超过 40%、连续删掉 3 段或 15 秒以上、"
          "删完后有口播稿要点在成片里消失（和稿子做 TF-IDF 比对；讲错重讲的那次不算，因为重讲还留着）。"
          "此时把原文清单给用户看，用户同意后原样回传 confirm_token 再调一次；"
          "token 绑定这个具体方案，改了保留段或输出路径就作废，不能自己编。"
          "想先看方案不动文件就用 dry_run=true。永不覆盖文件，撞名自动加序号。",
          {"path": _PATH,
           "keep_ids": {**_IDS, "description": "要保留的段编号"},
           "drop_ids": {**_IDS, "description": "要删掉的段编号"},
           "out": {"type": "string", "description": "可选，输出路径"},
           "bridge_s": {"type": "number", "description": "桥接阈值秒数，默认 0.5"},
           "dry_run": {"type": "boolean", "description": "只返回删留清单和比例，不出片"},
           "confirm_token": {"type": "string",
                             "description": "用户看过删除清单并同意后，原样回传上一次返回的 confirm_token"}},
          ["path"]),
])
