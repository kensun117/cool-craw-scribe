# **AI Agent 指令文档 (CoolScribe 项目 \- 单麦克风声纹识别版)**

## **👤 AI 角色设定**

你是一个资深的 Python 开发者，精通 discord.py / pycord、Apple Silicon 本地 AI 部署，以及基于 pyannote.audio 的声纹识别 (Speaker Diarization) 技术。

## **🎯 项目目标**

为 CoolTools 开发一个 Discord 会议机器人。

该机器人将在单麦克风场景下工作（频道内只有一个主收音设备）。录音结束后，需要先使用 AI 分离说话人，再使用本地模型转录文字。

## **🛠️ 技术栈与强制约束**

1. **Discord 框架**: pycord\[voice\]。  
2. **硬件环境**: Apple Silicon (Mac M系列芯片)。  
3. **转录引擎 (ASR)**: 必须使用 mlx-whisper 加速转录（严禁使用普通 openai-whisper）。  
4. **声纹引擎 (Diarization)**: 必须使用 pyannote.audio (版本 3.1)。  
5. **异步安全**: 转录和声纹推理都是重度阻塞操作，必须放入 asyncio.to\_thread 或 ThreadPoolExecutor 中执行。

## **⚙️ 核心处理工作流 (Workflow)**

录音结束后，生成一个总的音频文件 meeting.wav，然后在后台线程执行以下 Pipeline：

**Step 1: 声纹分离 (Pyannote)**

* 初始化 Pipeline.from\_pretrained("pyannote/speaker-diarization-3.1", use\_auth\_token=HF\_TOKEN)。  
* 注意：在 Mac 上，请将 pipeline 强制加载到 CPU 上运行以保证兼容性 pipeline.to(torch.device("cpu"))。  
* 获取各说话人的时间戳片段 (Turn, Track, Speaker)。

**Step 2: 语音转文字 (MLX-Whisper)**

* 调用 mlx\_whisper.transcribe(audio, path\_or\_hf\_repo="mlx-community/whisper-small-mlx", language="zh", word\_timestamps=True)。  
* 提取带有时间戳的文字片段 (Segments)。

**Step 3: 数据合并与对齐 (Merge)**

* 编写一个匹配算法：遍历 Whisper 输出的文字 segments，判断其时间戳落入 Pyannote 提供的哪个 Speaker 的时间段内。  
* 生成最终对话格式，如：  
  \[Speaker 1\] (00:00-00:05): 今天我们开个会。  
  \[Speaker 2\] (00:05-00:10): 好的，关于 Next.js 的部署...

## **📋 你的任务**

1. 在代码开头预留 HF\_TOKEN \= "YOUR\_HUGGINGFACE\_TOKEN" 环境变量位置。  
2. 编写 cogs/meeting\_recorder.py，实现上述完整的双模型 Pipeline 逻辑。  
3. 确保临时文件最终被完全清理。