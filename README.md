# CoolScribe

基于 pycord + pyannote.audio 3.1 + mlx-whisper 的 Discord 会议机器人（单麦克风场景）。

## 功能

- `/join_meeting`：机器人加入当前语音频道并开始录音
- `/leave_meeting`：停止录音，后台执行
  1. pyannote 声纹分离（CPU）
  2. MLX Whisper 转写（中文）
  3. 说话人与文本按时间戳对齐
- 输出格式：`[Speaker N] (00:00-00:05): 文本`
- 录音和处理中间文件自动清理

## 环境要求

- macOS (Apple Silicon)
- Python 3.10+
- `ffmpeg`

安装 ffmpeg:

```bash
brew install ffmpeg
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置

复制并填写环境变量：

```bash
cp .env.example .env
```

必须配置：

- `DISCORD_BOT_TOKEN`：Discord Bot Token
- `HF_TOKEN`：HuggingFace Token（需有 `pyannote/speaker-diarization-3.1` 访问权限）

## 启动

```bash
export $(grep -v '^#' .env | xargs)
python bot.py
```

## 本地录音测试（不连 Discord）

用于先验证麦克风采集和 `mlx-whisper` 识别效果：

```bash
python local_mic_test.py --seconds 12 --language zh
```

英文测试：

```bash
python local_mic_test.py --seconds 12 --language en
```

实时分段输出（边测边看到识别结果）：

```bash
python local_mic_test.py --stream --seconds 20 --chunk-seconds 4 --language zh
```

## 项目结构

```text
.
├── bot.py
├── cogs/
│   └── meeting_recorder.py
├── requirements.txt
└── .env.example
```
