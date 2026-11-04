"""Burned-in captions: Pillow renders a rounded translucent card, ffmpeg overlays it.

SRT / libass are not used. Rounded bars, padding, and per-frame width are easier
to guarantee by drawing the card ourselves. Timestamps come from a transcript of
the file being burned, so pass the finished cut, not the raw take.
"""

import json
import os
import re
import tempfile

from PIL import Image, ImageDraw, ImageFont

from agent.tools import edit, transcribe

FONT_CANDIDATES = [
    # macOS first, so a Mac keeps PingFang / Hiragino / Heiti.
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Linux CJK, then a Latin last resort.
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
TEXT_RGBA = (216, 210, 236, 255)   # 浅灰紫
BAR_RGBA = (24, 20, 30, 198)       # 深色半透明底条
LINE_RATIO = 1.42                  # 行高 = 字号 × 这个
WIDTH_RATIO = 0.88                 # 底条最宽占画面宽度的比例
FONT_DIVISOR = 17                  # 字号 ≈ 画面高度 ÷ 这个
MIN_FONT = 15
MAX_LINES = 3


def _font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise RuntimeError(f"找不到可用字体，试过：{FONT_CANDIDATES}")


def _wrap(text, font, max_w):
    """按画面宽度折行：英文优先在空格处断，中文逐字断。"""
    lines, cur = [], ""
    for ch in text.strip():
        trial = cur + ch
        if font.getlength(trial) <= max_w or not cur:
            cur = trial
            continue
        cut = cur.rfind(" ")
        # 空格靠后才值得回退断词，否则整行会被削掉一半
        if cut > len(cur) * 0.5:
            lines.append(cur[:cut].rstrip())
            cur = cur[cut + 1:] + ch
        else:
            lines.append(cur)
            cur = ch
    if cur.strip():
        lines.append(cur.strip())
    return lines


def _render_card(text, video_w, video_h, font_size):
    """画一张圆角半透明底条 + 文字的 PNG。"""
    max_w = video_w * WIDTH_RATIO
    pad_x, pad_y = round(font_size * 0.85), round(font_size * 0.5)
    font = _font(font_size)
    lines = _wrap(text, font, max_w - 2 * pad_x)
    while len(lines) > MAX_LINES and font_size > MIN_FONT:
        font_size = max(MIN_FONT, round(font_size * 0.88))
        font = _font(font_size)
        pad_x, pad_y = round(font_size * 0.85), round(font_size * 0.5)
        lines = _wrap(text, font, max_w - 2 * pad_x)
    lines = lines[:MAX_LINES]

    line_h = round(font_size * LINE_RATIO)
    text_w = max(font.getlength(ln) for ln in lines)
    card_w = min(round(text_w + 2 * pad_x), video_w)
    card_h = line_h * len(lines) + 2 * pad_y

    img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, card_w - 1, card_h - 1],
                           radius=min(round(card_h * 0.32), 26), fill=BAR_RGBA)
    for i, ln in enumerate(lines):
        x = (card_w - font.getlength(ln)) / 2
        draw.text((x, pad_y + i * line_h), ln, font=font, fill=TEXT_RGBA)
    return img


def burn_subtitles(path, out=None, font_size=None, language=None, bottom_margin=None):
    """给视频烧上字幕（有副作用：写新文件）。

    时间戳一律用这个文件自己的转录结果——所以要传剪好的成片，不是原始素材，
    否则字幕会和画面对不上。
    """
    src = edit._require(path)
    info = edit.probe(src)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if not video:
        raise RuntimeError(f"{os.path.basename(src)} 没有画面轨，烧不了字幕")
    if not info["has_audio"]:
        raise RuntimeError(f"{os.path.basename(src)} 没有音轨，无法转录出字幕时间戳")
    w, h = int(video["width"]), int(video["height"])

    # 字幕要的是拼写对：用口播稿当词汇提示，专有名词才不会听错（成片已剪干净，不怕吞口头禅）
    segments = [s for s in transcribe.transcribe(src, language=language,
                                                 vocabulary_hint=True)["segments"]
                if s["text"].strip()]
    if not segments:
        raise RuntimeError("转录结果是空的，没有可烧的字幕")

    size = int(font_size) if font_size else max(MIN_FONT, round(h / FONT_DIVISOR))
    margin = int(bottom_margin) if bottom_margin else round(h * 0.06)
    dst = edit._pick_out(edit._out_path(src, "_sub", out), src)

    workdir = tempfile.mkdtemp(prefix="sub_")
    try:
        cards = []
        for i, seg in enumerate(segments):
            png = os.path.join(workdir, f"card{i:03d}.png")
            img = _render_card(seg["text"], w, h, size)
            img.save(png)
            cards.append((png, img.size, seg))

        cmd = [edit.FFMPEG, "-y", "-v", "error", "-i", src]
        for png, _, _ in cards:
            cmd += ["-i", png]
        steps, cur = [], "0:v"
        for i, (_, (cw, ch), seg) in enumerate(cards, start=1):
            nxt = f"v{i}"
            steps.append(
                f"[{cur}][{i}:v]overlay=x=(W-w)/2:y=H-h-{margin}:"
                f"enable='between(t,{seg['start']:.3f},{seg['end']:.3f})'[{nxt}]")
            cur = nxt
        cmd += ["-filter_complex", ";".join(steps), "-map", f"[{cur}]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-c:a", "copy", dst]
        edit._run(cmd)
    finally:
        for f in os.listdir(workdir):
            os.unlink(os.path.join(workdir, f))
        os.rmdir(workdir)

    edit._record(dst, "burn_subtitles", [src],
                 {"segments": len(segments), "font_size": size,
                  "video": f"{w}x{h}", "margin": margin})
    return {"path": dst, "duration_s": edit._duration(dst), "segments": len(segments),
            "font_size": size, "video_size": f"{w}x{h}",
            "max_bar_width_px": max(c[1][0] for c in cards),
            "verified_by_ffprobe": True}


def preview_subtitle_card(text, video_width=1280, video_height=720, font_size=None,
                          out=None):
    """只渲染一张字幕图片来看效果，不碰视频。用来先确认字号和折行。"""
    size = int(font_size) if font_size else max(MIN_FONT, round(int(video_height) / FONT_DIVISOR))
    img = _render_card(text, int(video_width), int(video_height), size)
    dst = os.path.abspath(os.path.expanduser(
        out or os.path.join(tempfile.gettempdir(), "subtitle_preview.png")))
    img.save(dst)
    return {"path": dst, "card_size": f"{img.size[0]}x{img.size[1]}",
            "video_size": f"{video_width}x{video_height}", "font_size": size,
            "fits_in_frame": img.size[0] <= int(video_width)}


def _tool(name, fn, description, properties, required):
    return name, {
        "fn": lambda **kw: json.dumps(fn(**kw), ensure_ascii=False),
        "schema": {"name": name, "description": description,
                   "input_schema": {"type": "object", "properties": properties,
                                    "required": required}},
    }


TOOLS = dict([
    _tool("burn_subtitles", burn_subtitles,
          "给视频烧上字幕（深色圆角半透明底条 + 大号浅灰紫字），写新文件。"
          "字幕时间戳来自对这个文件自己的转录，所以必须传剪好的成片，不能传原始素材，"
          "否则剪掉的部分会让字幕整体错位。行宽按画面宽度算，小分辨率也不会溢出。",
          {"path": {"type": "string", "description": "剪好的成片绝对路径"},
           "out": {"type": "string", "description": "可选，输出路径"},
           "font_size": {"type": "integer", "description": "可选，默认按画面高度 ÷17 自动定"},
           "language": {"type": "string", "description":
                        "可选，转录语言。不确定就别传，让引擎自己识别——"
                        "不要照用户提问用的语言来猜音频语言，指定错了转录质量会变差"},
           "bottom_margin": {"type": "integer", "description": "可选，底边距像素"}},
          ["path"]),
    _tool("preview_subtitle_card", preview_subtitle_card,
          "先渲染一张字幕图看效果（不碰视频），确认字号、折行、是否溢出。",
          {"text": {"type": "string", "description": "示例字幕文字"},
           "video_width": {"type": "integer"}, "video_height": {"type": "integer"},
           "font_size": {"type": "integer"},
           "out": {"type": "string", "description": "可选，PNG 输出路径"}},
          ["text"]),
])
