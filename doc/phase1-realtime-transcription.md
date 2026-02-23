# Phase 1: 实时语音转录 + 声纹识别

> 状态：已完成（voice_bot.py）

## 目标

用户在 Discord 语音频道开会时，机器人实时将语音转为文字，通过声纹识别标注说话人，转录结果显示在语音频道的文字聊天中。会议结束时生成完整的会议原始纪要。

---

## 当前实现：voice_bot.py

### 入口文件

```
voice_bot.py       # 独立运行，不依赖 bot.py / cogs 架构
```

运行方式：
```bash
source .venv/bin/activate
python voice_bot.py
```

环境变量（`.env`，启动时自动加载）：
```
VOICE_DISCORD_TOKEN=...      # Discord Bot Token
HF_TOKEN=...                 # HuggingFace Token（声纹模型下载）
OPENCLAW_GATEWAY_URL=...     # 可选，Phase 2 集成用，默认 http://localhost:18789
OPENCLAW_GATEWAY_TOKEN=...   # 可选
```

---

## Discord 命令

| 命令 | 参数 | 说明 |
|------|------|------|
| `/voice_on` | `language`（zh/en，默认 zh） | 加入语音频道，开始实时转录 |
| `/voice_off` | 无 | 停止转录，生成并发送会议纪要 |
| `/register_voice` | `seconds`（5-30，默认 10） | 录制声纹样本，注册说话人身份 |

---

## 功能说明

### 1. VAD 切分（按停顿切句）

不再按固定时长（原来3秒）切断，而是根据 Silero VAD 的停顿自动判断句子边界：

```
有人说话 → 持续累积音频
停顿 >= SILENCE_TIMEOUT → 认为一句话说完 → 送去转录
累积 >= MAX_SPEECH_DURATION → 强制切断，防止单句过长
```

**可调参数（`voice_bot.py` 顶部）：**

| 常量 | 默认值 | 说明 |
|------|--------|------|
| `VAD_FRAME_DURATION` | 0.2 秒 | VAD 检测间隔 |
| `SILENCE_TIMEOUT` | 0.8 秒 | 停顿多久算说完一句 |
| `MAX_SPEECH_DURATION` | 15.0 秒 | 单句最长时长，超过强制切断 |
| `VAD_THRESHOLD` | 0.5 | Silero VAD 判断有声音的概率阈值（0~1） |

说话节奏较慢时可将 `SILENCE_TIMEOUT` 调大到 `1.2`；想要更快出结果可调小到 `0.5`。

### 2. 声纹识别

**注册流程：**
1. 用户执行 `/register_voice seconds:15`
2. 机器人录制指定时长语音
3. `speechbrain/ecapa-tdnn` 模型提取 speaker embedding（在 CPU 上运行）
4. embedding 以 `.npy` 格式持久化到 `data/speaker_profiles/`

**识别流程（每句话转录前）：**
1. 对当前音频提取 embedding
2. 与所有已注册声纹做余弦相似度对比
3. 相似度 >= 0.7 → 匹配成功，使用注册名
4. 未匹配 → 回退到 Discord 显示名

**声纹存储结构：**
```
data/speaker_profiles/
├── profiles.json           # 元数据索引
└── embeddings/
    ├── 123456789.npy       # 用户 Discord ID 命名
    └── 987654321.npy
```

**profiles.json 格式：**
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

机器人启动时自动加载已有声纹，重启不丢失数据。

### 3. ASR 转录

- 引擎：`mlx-whisper`（Apple Silicon 专用，比 openai-whisper 快 3-5 倍）
- 模型：`mlx-community/whisper-small-mlx`（244M 参数，速度与精度平衡）
- 重采样：`torchaudio.transforms.Resample`（sinc 算法，48kHz → 16kHz）

### 4. 实时输出格式

每句话说完后，立即发到 Discord 文字频道：
```
🎙️ [张三]: 我觉得部署方案可以用 Vercel
🎙️ [李四]: 那数据库怎么处理？
```

### 5. 会后纪要（/voice_off 触发）

停止转录后，对每个用户会议期间所有有声音帧（经 VAD 过滤的 PCM）重新跑一遍 mlx-whisper，生成高质量完整转录：

```
--- 会议纪要（完整录音转录版）---
时长: 12 分 34 秒
参与者: 张三, 李四

【张三】
我觉得部署方案可以用 Vercel，数据库用 Supabase 就行，成本低而且免运维。

【李四】
那边的 API 限速怎么处理？我们上次遇到过这个问题。
```

> 会后纪要使用完整录音重新转录，准确度高于实时片段拼接。只有语音帧（VAD 有声）会被送入 Whisper，静音段已过滤，避免 Whisper 产生幻觉（重复循环文字）。

---

## 音频处理流程

```
Discord 音频流（48kHz 立体声 PCM）
  ↓
读指针增量读取（不 truncate，消除读写竞争）
  ↓
每 200ms 取一帧（VAD frame）
  ↓
Silero VAD 判断是否有人声
  ├── 有声 → 追加到 speech buffer + full_audio（用于会后纪要）
  └── 静音 >= 0.8s → 送去实时转录
            ↓
      stereo → mono
            ↓
      48kHz → 16kHz（torchaudio Resample）
            ↓
      soundfile 写 WAV（临时文件）
            ↓
      speechbrain/ecapa-tdnn 声纹匹配（asyncio.to_thread）
            ↓
      mlx-whisper 转录（asyncio.to_thread）
            ↓
      发送到 Discord 文字频道
            ↓
/voice_off → full_audio（各用户有声帧合集）→ mlx-whisper 完整转录 → 会议纪要
```

---

## 技术约束

1. **ASR 引擎**：必须使用 `mlx-whisper`，禁止使用普通 `openai-whisper`
2. **声纹模型**：`speechbrain/ecapa-tdnn`，通过 pyannote 的 `PretrainedSpeakerEmbedding` 加载
3. **硬件**：Apple Silicon，声纹模型强制 `torch.device("cpu")`
4. **异步安全**：转录和声纹推理均通过 `asyncio.to_thread` 在线程池中执行，不阻塞事件循环

---

## 不在 Phase 1 范围

- OpenClaw 集成（触发词检测、命令确认、AI 回复） → Phase 2
- 多服务器声纹隔离（当前所有服务器共享声纹库）
- 录音文件持久化保存（处理完即删除临时文件）
- 声纹管理命令（列出已注册声纹、删除声纹等）
