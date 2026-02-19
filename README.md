# CoolScribe

Discord 会议语音机器人 — 实时转录、声纹识别、AI 辅助会议。

## 项目定位

CoolScribe 是 CoolTools 生态中的会议效率工具。它加入 Discord 语音频道，实时将语音转为文字，通过声纹识别区分说话人，生成结构化的会议原始纪要。后续将与 OpenClaw 集成，实现会议中的 AI 辅助查询和事实核查。

## 实现分期

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 实时语音转录 + 声纹识别 + 持久化 | 开发中 |
| Phase 2 | OpenClaw 会议交互（语音触发、确认执行） | 待实现 |

详细规格文档：
- Phase 1: [doc/phase1-realtime-transcription.md](doc/phase1-realtime-transcription.md)
- Phase 2: [doc/phase2-openclaw-interaction.md](doc/phase2-openclaw-interaction.md)

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| Discord 框架 | pycord[voice] | 语音频道连接与录音 |
| 语音转文字 (ASR) | mlx-whisper | Apple Silicon 加速，禁用普通 openai-whisper |
| 声纹识别 | pyannote.audio 3.1 + speechbrain/ecapa-tdnn | 说话人分离与识别 |
| 语音活动检测 | Silero VAD | 过滤静音片段 |
| 硬件环境 | Apple Silicon (Mac M 系列) | pyannote 强制 CPU 运行 |

## 环境要求

- macOS (Apple Silicon)
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

## 快速开始

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填写 DISCORD_BOT_TOKEN 和 HF_TOKEN

# 启动
export $(grep -v '^#' .env | xargs)
python bot.py
```

## 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `DISCORD_BOT_TOKEN` | 是 | Discord Bot Token |
| `HF_TOKEN` | 是 | HuggingFace Token（需有 pyannote/speaker-diarization-3.1 访问权限） |
| `LOG_LEVEL` | 否 | 日志级别，默认 INFO |
| `OPENCLAW_GATEWAY_URL` | 否 | OpenClaw Gateway WebSocket 地址（Phase 2） |
| `OPENCLAW_GATEWAY_TOKEN` | 否 | OpenClaw Gateway Token（Phase 2） |

## 项目结构

```
.
├── bot.py                  # 入口，加载 cogs
├── cogs/
│   ├── meeting_recorder.py # 会后批量转录（声纹分离 + 转录 + 合并）
│   └── realtime_voice.py   # 实时语音转录（Phase 1 重构目标）
├── data/
│   └── speaker_profiles/   # 声纹持久化存储（Phase 1 新增）
├── doc/
│   ├── phase1-realtime-transcription.md
│   └── phase2-openclaw-interaction.md
├── local_mic_test.py       # 本地麦克风测试工具
├── requirements.txt
└── .env.example
```

## 本地测试（不连 Discord）

`local_mic_test.py` 用于在不连接 Discord 的情况下测试麦克风采集和 mlx-whisper 转录效果。排查问题时优先用此工具验证本地环境是否正常。

```bash
source .venv/bin/activate

# 基础测试：录 12 秒，转录中文，输出带时间戳的分段结果
python local_mic_test.py --seconds 12 --language zh

# 英文测试
python local_mic_test.py --seconds 12 --language en

# 实时流式测试：每 4 秒输出一次转录，共 20 秒，模拟实时效果
python local_mic_test.py --stream --seconds 20 --chunk-seconds 4 --language zh
```

**工作原理：**
- 使用 `sounddevice` 从系统麦克风录音（16kHz, mono, float32）
- 基础模式：录完后一次性调用 `mlx_whisper.transcribe()` 输出分段结果
- 流式模式：按 `--chunk-seconds` 分段录音，每段录完立即转录输出
- 临时文件自动清理

**验证要点：**
- mlx-whisper 模型能正常加载并输出结果
- 系统麦克风权限和音频采集正常
- 流式模式下分段转录的延迟和准确度是否可接受
