# 口播全流程助手

Talking-Head Full Pipeline Assistant. 跑在本机的口播助手：按主题写口播稿、用 OBS 录口播、提词器跟读，录完自动剪掉冷场和讲错的地方，再烧上字幕。

聊天页、工具循环、语料和成片都留在本机。对外只有两处：你配置的大模型（Anthropic Messages API，或兼容网关），以及你明确要求抓取的网页。

## 功能

- **口播稿。** 把文章抓进本地语料库，用字符级 TF-IDF 检索（中英混排），再写成口语稿。段落末尾可以标 `[来源:文件名]`。检索结果里没有的事实不应编造。
- **提词器。** 稿子推到全屏页 `/teleprompter`，可滚动、暂停、回顶。macOS 上开录时会把这个页面切到前台。
- **OBS 录制。** 通过 obs-websocket 5.x 控制场景、输入源、滤镜、开录和停录。设备 id 和 kind 以 OBS 的枚举为准。停录后用 ffprobe 核对时长、画面和音轨；音轨是数字静音时会提醒。
- **自动剪辑。** 可以按静音剪，也可以按转录段编号剪。模型只报段编号，时间戳由工具查表。删得太多、连续大段删除、或会让稿子要点从成片里消失时，先给确认口令，同意后才出片。成片永远是新文件。`edit_history` 可以回溯到原片。
- **字幕。** Pillow 画出圆角半透明底条，ffmpeg 叠进画面。时间戳来自「当前这个文件」的转录，所以烧的是剪好的成片，不是原始素材。macOS 优先用 PingFang / 冬青黑体 / 华文黑体；Linux 回退到文泉驿微米黑、Noto Sans CJK 或 Droid Sans Fallback。
- **对话窗口。** 中文单页，SSE 流式显示正文和每一次工具调用，中途可以停止。已经做完的步骤不会被撤销。

录制相关的本机集成（拉起 OBS、提词器置前、`caffeinate` 防睡眠、睡眠唤醒通知）按 **macOS** 实现。对话服务、剪辑、转录和检索本身只要有 Python 3.12 和 ffmpeg 就能跑。

## 目录结构

```
.
  LICENSE                       # MIT
  README.md
  requirements.txt              # 运行依赖
  .env.example                  # 环境变量模板，复制成 .env
  .gitignore
  env.sh                        # 加载 .env，并做旧变量名映射
  preflight.py                  # 环境检查
  .claude/
    launch.json                 # 本地启动：助手服务、preflight
  agent/
    __init__.py
    server.py                   # 对话窗口 + agent 循环
    llm.py                      # Anthropic Messages 客户端
    tools/
      __init__.py
      obs.py                    # 裸 obs-websocket 协议
      edit.py                   # ffmpeg 剪辑 + 静音检测
      transcribe.py             # 转录 + 按段选剪
      research.py               # 语料库 + 检索
      subtitle.py               # 字幕卡片，ffmpeg 烧进成片
    web/
      index.html                # 聊天页
      teleprompter.html         # 提词器
  docs/
    message-flow.md             # 聊天请求与 SSE
    obs-control.md              # OBS 协议与工具
    diagrams/
      agent-message-flow.html   # 消息流示意图
```

运行后还会出现这些路径，它们不进版本库：`.venv/`、`.env`、`corpus/`（语料）、录像和剪辑中间文件。

请求怎么从浏览器走到模型和工具，见 [docs/message-flow.md](docs/message-flow.md)。OBS 客户端见 [docs/obs-control.md](docs/obs-control.md)。

## 环境要求

- Python 3.12 或更高
- [ffmpeg](https://ffmpeg.org/) 和 `ffprobe`，在 `PATH` 里
- [OBS Studio](https://obsproject.com/) 28 或更高，并打开内置 WebSocket（工具 → WebSocket 服务器设置）
- 大模型凭据：`ANTHROPIC_API_KEY`，或网关用的 `ANTHROPIC_AUTH_TOKEN`
- 转录引擎（按段剪辑和烧字幕时才需要），装一个即可：
  - Apple Silicon：`mlx-whisper`
  - 通用：`faster-whisper`
  - 或 `openai-whisper`

## 环境配置

项目固定用 Python 3.12。系统自带的 `python3` 若更旧，不要用它装依赖，单独建虚拟环境。

```bash
cd 项目目录
uv venv --python 3.12 .venv
# 没有 uv 时：python3.12 -m venv .venv

.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install faster-whisper
# Apple Silicon 可改为：.venv/bin/python -m pip install mlx-whisper

cp .env.example .env
```

编辑 `.env`。程序只读 `ANTHROPIC_*` 和下面这些 OBS / 运行变量。

| 变量 | 必填 | 作用 |
|---|---|---|
| `ANTHROPIC_API_KEY` | 二选一 | 直连 Anthropic，请求头 `x-api-key` |
| `ANTHROPIC_AUTH_TOKEN` | 二选一 | 兼容网关，请求头 `Authorization: Bearer`。两个都填时用这个 |
| `ANTHROPIC_BASE_URL` | 否 | 默认 `https://api.anthropic.com`，不要带路径 |
| `ANTHROPIC_MODEL` | 否 | 默认 `claude-sonnet-4-20250514` |
| `OBS_PASSWORD` | OBS 开了认证就必填 | WebSocket 密码 |
| `OBS_URL` | 否 | 默认 `ws://localhost:4455` |
| `OBS_PORT` | 否 | 没设 `OBS_URL` 时，`preflight.py` 用它，默认 `4455` |
| `PORT` | 否 | 本机服务端口，默认 `8788` |
| `CORPUS_DIR` | 否 | 语料目录，默认 `./corpus` |
| `TRANSCRIBE_MODEL` | 否 | Whisper 规格，默认 `small` |
| `FFMPEG` / `FFPROBE` | 否 | 可执行文件名或绝对路径 |

旧环境如果只写了 `PARALLIGHT_BASE_URL` 和 `PARALLIGHT_API_KEY`，仍然可用。`source env.sh` 仅在对应的 `ANTHROPIC_*` 为空时，把它们映射成 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_AUTH_TOKEN`。已经写好的 `ANTHROPIC_*` 不会被盖掉。新的 `.env` 请直接用 `ANTHROPIC_*`。

**每次开新终端先执行：**

```bash
cd 项目目录 && source env.sh && .venv/bin/python <脚本>
```

- `source env.sh`：读入 `.env`，按上面的规则处理 `PARALLIGHT_*` 旧名，并把 Homebrew 的 ffmpeg（Apple Silicon `/opt/homebrew/bin`，Intel `/usr/local/bin`）以及 `.venv/bin` 加进 `PATH`。环境变量不跨终端，每次都要重新 source。脚本不打印密钥。
- `.venv/bin/python`：固定走项目虚拟环境里的 Python 3.12，不跟系统 `python3` 抢。

**检查环境是否就绪：**

```bash
cd 项目目录 && source env.sh && .venv/bin/python preflight.py
```

全过会打印聊天页和提词器地址。OBS 没开、或还没写密钥时，对应项失败，按提示补上再跑。没装转录引擎只是警告，剪辑和烧字幕之前再装即可。

## 安装与运行

环境配好之后：

```bash
cd 项目目录 && source env.sh && .venv/bin/python preflight.py
cd 项目目录 && source env.sh && .venv/bin/python -m agent.server
```

- 聊天页：<http://localhost:8788/>
- 提词器：<http://localhost:8788/teleprompter>

服务只监听 `127.0.0.1`。换端口：

```bash
cd 项目目录 && source env.sh && PORT=8790 .venv/bin/python -m agent.server
```

## 使用

在聊天页里按这个顺序说即可。

1. 给出主题和文章 URL。助手抓取正文（`fetch_article`）、检索（`corpus_search`），写成带出处的短口播稿。
2. 稿子推上提词器（`teleprompter_load`）。页面没开就自己打开 `/teleprompter`。不要用整屏采集，否则提词器会进画面；窗口采集或应用采集可以，只要被采集的不是提词器那个窗口。
3. 让它开录。它会核对 OBS 再开始；macOS 上若提词器已有稿子，会把页面切到前台。聊天页或提词器上的按钮都可以停录，返回 `outputPath`。麦克风没信号会马上说明。
4. 让它收紧成片。冷场是 `detect_silences` 再 `cut_spans`。按内容剪是 `transcribe` 再 `cut_segments`：删掉的原文会逐条列出来；触发拦截时带回 `confirm_token`，你同意后再出片。
5. 对**剪好的成片**烧字幕（`burn_subtitles`）。想先看字号和折行，用 `preview_subtitle_card`。

如果要判断这份语料值不值得上向量库，让它跑 `retrieval_experiment`。报告里的 `lsa` 是对本语料 TF-IDF 做的降维，不是训练过的向量模型。

## 许可证

[MIT](LICENSE)。
