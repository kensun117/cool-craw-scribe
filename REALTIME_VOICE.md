# 实时语音聊天功能

此分支实现了 Discord 语音频道的实时转录与 OpenClaw 集成功能。

## 新增功能

### 1. 实时语音转录
- 边说话边转录，无需等待会议结束
- 使用 Silero VAD 进行语音活动检测
- 每 2 秒处理一个音频片段
- 自动区分说话人并标记

### 2. OpenClaw 集成
- 通过 WebSocket 连接到 OpenClaw Gateway
- 实时将转录文本发送给 OpenClaw
- 接收 OpenClaw 的回复并发送到 Discord 文字频道

### 3. 声纹识别
- 使用 pyannote 提取说话人 embedding
- 注册后可持续识别同一用户
- 支持多说话人区分

## 新增命令

| 命令 | 说明 |
|------|------|
| `/start_voice_chat` | 启动实时语音聊天模式 |
| `/stop_voice_chat` | 停止实时语音聊天 |
| `/register_voice` | 注册用户声纹（10秒录音） |
| `/set_openclaw_channel` | 设置 OpenClaw 回复的目标频道 |

## 配置

在 `.env` 文件中添加：

```bash
# OpenClaw Gateway 配置
OPENCLAW_GATEWAY_URL=ws://localhost:8080/ws
OPENCLAW_GATEWAY_TOKEN=your_token_here
```

## 使用方法

1. 进入语音频道
2. 执行 `/register_voice` 注册声纹（可选，但推荐）
3. 执行 `/start_voice_chat` 启动
4. 开始说话，转录会实时显示在文字频道
5. OpenClaw 会自动回复转录的内容
6. 执行 `/stop_voice_chat` 停止

## 技术细节

- 音频采样率：Discord 48kHz → Whisper 16kHz
- 转录模型：mlx-community/whisper-small-mlx
- 声纹模型：speechbrain/ecapa-tdnn
- VAD 模型：Silero VAD
