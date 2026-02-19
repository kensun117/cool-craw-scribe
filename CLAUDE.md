# AI Agent 指令文档 (CoolScribe 项目)

## 角色设定

你是一个资深的 Python 开发者，精通 pycord、Apple Silicon 本地 AI 部署、pyannote.audio 声纹识别和 mlx-whisper 语音转录。

## 项目目标

为 CoolTools 开发一个 Discord 会议语音机器人。机器人加入语音频道，**实时**将语音转为文字，通过声纹识别区分说话人，生成会议原始纪要。后续将与 OpenClaw 集成，实现会议中的 AI 辅助。

项目分两个阶段：
- **Phase 1**: 实时语音转录 + 声纹识别 + 持久化（当前重点）
- **Phase 2**: OpenClaw 会议交互（语音触发词 → 确认执行 → 结果回显）

详细规格见 `doc/phase1-realtime-transcription.md` 和 `doc/phase2-openclaw-interaction.md`。

## 技术栈与强制约束

1. **Discord 框架**: `pycord[voice]`，使用 cog 架构（`bot.py` + `cogs/`）。
2. **硬件环境**: Apple Silicon (Mac M 系列芯片)。
3. **转录引擎 (ASR)**: 必须使用 `mlx-whisper`（严禁使用普通 `openai-whisper`）。
4. **声纹引擎**: `pyannote.audio` 3.1（声纹分离）+ `speechbrain/ecapa-tdnn`（说话人 embedding）。
5. **VAD**: Silero VAD，用于过滤静音片段。
6. **异步安全**: 转录和声纹推理是阻塞操作，必须放入 `asyncio.to_thread` 或 `ThreadPoolExecutor`。
7. **pyannote 兼容性**: 在 Mac 上必须 `pipeline.to(torch.device("cpu"))`。

## 核心工作流

### 实时转录模式（Phase 1 重点）

```
用户说话 → Discord 音频流 → 内存 buffer（2秒）
    → Silero VAD 过滤静音
    → 48kHz→16kHz 重采样
    → mlx-whisper 转录
    → 声纹匹配识别说话人
    → 发送到语音频道文字聊天: [说话人名]: 转录文字
```

### 会后批量模式（已实现）

```
录音结束 → meeting.wav
    → Step 1: pyannote 声纹分离（CPU）
    → Step 2: mlx-whisper 转录
    → Step 3: 时间戳对齐合并
    → 输出: [Speaker N] (MM:SS-MM:SS): 文字
```

## 项目结构

```
.
├── bot.py                          # 入口，加载 cogs
├── cogs/
│   ├── meeting_recorder.py         # 会后批量转录 cog
│   └── realtime_voice.py           # 实时转录 cog（Phase 1 重构目标）
├── data/
│   └── speaker_profiles/           # 声纹持久化存储
│       ├── profiles.json
│       └── embeddings/*.npy
├── doc/
│   ├── phase1-realtime-transcription.md
│   └── phase2-openclaw-interaction.md
├── local_mic_test.py               # 本地麦克风测试（不连 Discord）
├── requirements.txt
├── .env.example
└── CLAUDE.md
```

## 测试

- 本地测试工具 `local_mic_test.py` 可在不连接 Discord 的情况下验证麦克风和 mlx-whisper 环境，使用方式见 `README.md`
- 排查转录问题时优先使用本地测试工具，排除 Discord 音频流的干扰

## 编码规范

- 环境变量通过 `os.getenv()` 读取，`HF_TOKEN` 等敏感信息不要硬编码
- Discord slash command 使用 pycord 的 `@commands.slash_command` 装饰器
- 临时文件必须在 `finally` 块中清理（`shutil.rmtree`）
- 日志使用 `logging.getLogger(__name__)`
