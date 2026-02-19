# Phase 1: 实时语音转录 + 声纹识别

> 状态：开发中

## 目标

用户在 Discord 语音频道开会时，机器人实时将语音转为文字，通过声纹识别标注说话人，转录结果显示在语音频道的文字聊天中。会议结束时生成完整的会议原始纪要。

## 功能清单

### 1. 实时语音转录

**用户操作：**
- `/start_voice_chat [language]` — 机器人加入语音频道，开始实时转录
- `/stop_voice_chat` — 停止转录，机器人离开频道，输出完整纪要

**行为：**
- 机器人加入用户所在的语音频道
- 持续接收音频，每 2 秒处理一个片段
- 通过 Silero VAD 检测语音活动，跳过静音
- 使用 mlx-whisper 转录语音为文字
- 转录结果实时发送到**语音频道的文字聊天**中
- 支持中文 (`zh`) 和英文 (`en`)

**输出格式：**
```
[张三]: 我觉得部署方案可以用 Vercel
[李四]: 那数据库怎么处理？
[张三]: 用 Supabase 就行
```

**不带时间戳** — 时间戳在实时场景下意义不大，Discord 消息本身有时间。会后纪要中也不需要精确到秒的时间戳，重点是"谁说了什么"。

### 2. 声纹注册与识别

**用户操作：**
- `/register_voice [seconds]` — 录制 5-30 秒语音样本，注册声纹

**行为：**
- 用户执行命令后，机器人录制指定时长的语音
- 使用 speechbrain/ecapa-tdnn 提取 speaker embedding
- 将 embedding 与用户的 Discord ID、用户名绑定存储
- 后续转录时，通过余弦相似度匹配识别说话人
- 未注册用户显示 Discord 用户名（从频道成员信息获取）

**识别逻辑：**
- 相似度阈值：0.7（高于此值认为匹配成功）
- 未匹配到已注册声纹时，回退到 Discord 用户名

### 3. 声纹持久化

**存储位置：** `data/speaker_profiles/`

**存储格式：**
```
data/speaker_profiles/
├── profiles.json           # 元数据索引
└── embeddings/
    ├── 123456789.npy       # 用户 ID 命名的 embedding 文件
    └── 987654321.npy
```

**profiles.json 结构：**
```json
{
  "123456789": {
    "user_id": 123456789,
    "username": "张三",
    "registered_at": 1700000000.0,
    "embedding_file": "embeddings/123456789.npy"
  }
}
```

**行为：**
- 注册声纹时写入文件
- 机器人启动时自动加载已有声纹
- 重启不丢失数据

### 4. 会后纪要生成

**触发时机：** 用户执行 `/stop_voice_chat` 时

**行为：**
- 汇总本次会议所有转录记录
- 按时间顺序整理为完整 transcript
- 合并同一说话人的连续短句（避免碎片化）
- 发送到语音频道的文字聊天中

**输出格式：**
```
--- 会议纪要 ---
时长: 45 分钟
参与者: 张三, 李四, 王五

[张三]: 今天讨论部署方案
[李四]: 我建议用 Vercel，前端部署很方便
[张三]: 那后端呢？
[王五]: 后端可以用 Docker 部署到 fly.io
...
```

## 现有代码问题与改造方向

### 问题 1: 非真正实时

当前 `realtime_voice.py` 使用 `discord.sinks.WaveSink`，这是一个批量录音 Sink —— 调用 `stop_recording()` 后才能拿到音频数据。当前代码通过反复 stop/start 来模拟实时，但这不可靠。

**改造方向：** 实现自定义 Sink，在 `write()` 方法中直接将音频数据推入内存 buffer，不依赖 stop_recording 触发。或者调研 pycord 是否提供 per-packet callback 机制。

### 问题 2: 音频重采样质量

当前使用简单线性插值从 48kHz 降采样到 16kHz，可能引入音质损失。

**改造方向：** 使用 `torchaudio.transforms.Resample` 或 `scipy.signal.resample`，质量更好。

### 问题 3: 声纹数据不持久

当前 `SpeakerRecognizer` 的 `speaker_profiles` 是内存 dict，重启丢失。

**改造方向：** 按上述存储格式持久化到 `data/speaker_profiles/`。

### 问题 4: 代码重复

`voice_bot.py` 与 `cogs/realtime_voice.py` 功能重叠，应删除 `voice_bot.py` 和 `start_voice_bot.sh`，统一用 cog 架构。

### 问题 5: 未使用的依赖

`requirements.txt` 中 `websocket-client`、`speechrecognition`、`aiofiles` 未被使用，应清理。

## 技术约束（来自 CLAUDE.md）

1. **ASR 引擎**：必须使用 `mlx-whisper`，禁止使用普通 `openai-whisper`
2. **Diarization 引擎**：必须使用 `pyannote.audio` 3.1
3. **硬件**：Apple Silicon，pyannote pipeline 强制 `torch.device("cpu")`
4. **异步安全**：转录和声纹推理是阻塞操作，必须放入 `asyncio.to_thread` 或 `ThreadPoolExecutor`

## 不在 Phase 1 范围

- OpenClaw 集成（触发词检测、命令确认、AI 回复） → Phase 2
- Web 实时显示页面
- 多服务器声纹隔离（当前所有服务器共享声纹库）
- 录音文件持久化保存（当前处理完即删除）
