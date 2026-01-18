<a href="https://trendshift.io/repositories/9828" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9828" alt="dnhkng%2FGlaDOS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# GLaDOS Personality Core

> *"The Enrichment Center reminds you that the Weighted Companion Cube will never threaten to stab you."*

GLaDOS (Genetic Lifeform and Disk Operating System) is the AI antagonist from Valve's Portal series—a sardonic, passive-aggressive superintelligence who views humans as test subjects worthy of both study and mockery.

This project brings her to life on real hardware. She sees through a camera, hears through a microphone, speaks through a speaker, and judges you accordingly.

[Join our Discord!](https://discord.com/invite/ERTDKwpjNB) | [Sponsor the project](https://ko-fi.com/dnhkng)

https://github.com/user-attachments/assets/c22049e4-7fba-4e84-8667-2c6657a656a0

## Vision

Most voice assistants wait for wake words. GLaDOS doesn't wait—she observes, thinks, and speaks when she has something to say.

**Goals:**
- **Proactive behavior**: React to events (vision, sound, time) without being prompted
- **Emotional state**: PAD model (Pleasure-Arousal-Dominance) for reactive mood
- **Persistent personality**: HEXACO traits provide stable character across sessions
- **Multi-agent architecture**: Subagents handle research, memory, emotions; main agent stays focused
- **Real-time conversation**: Optimized latency, natural interruption handling

## What's New

- **Vision**: FastVLM gives her eyes. [Details](/vision.md) | [Demo](https://www.youtube.com/watch?v=JDd9Rc4toEo)
- **Autonomy**: She watches, waits, and speaks when she has something to say. [Details](/autonomy.md)
- **MCP Tools**: Extensible tool system for home automation, system info, etc. [Details](/mcp.md)
- **8GB SBC**: Runs on a Rock5b with RK3588 NPU. [Branch](https://github.com/dnhkng/RKLLM-Gradio)

## Roadmap

- [x] Train GLaDOS voice
- [x] Personality that actually sounds like her
- [x] Vision via VLM
- [x] Autonomy (proactive behavior)
- [x] MCP tool system
- [ ] Emotional state (PAD model)
- [ ] Long-term memory
- [ ] Observer agent (behavior adjustment)
- [ ] 3D-printable enclosure
- [ ] Animatronics

## Architecture

```mermaid
flowchart TB
    subgraph Input
        mic[🎤 Microphone] --> vad[VAD] --> asr[ASR]
        text[⌨️ Text Input]
        tick[⏱️ Timer]
        cam[📷 Camera]
    end

    subgraph Minds["Subagents"]
        weather[Weather]
        emotion[Emotion]
        news[News]
        memory[Memory]
    end

    ctx[📋 Context]

    subgraph Core["Main Agent"]
        llm[🧠 LLM]
        tts[TTS]
    end

    subgraph Output
        speaker[🔊 Speaker]
        images[🖼️ Images]
        motors[⚙️ Animatronics]
    end

    asr -->|priority| llm
    text -->|priority| llm
    tick --> Minds
    cam --> Minds
    tick -->|autonomy| llm

    Minds -->|write| ctx
    ctx -->|read| llm
    llm --> tts --> speaker
    llm <-->|MCP| tools[Tools]
    tools --> images
    tools --> motors
```

GLaDOS runs a loop: each tick she reads her slots (weather, news, vision, mood), decides if she has something to say, and speaks. No wake word—if she has an opinion, you'll hear it.

**Two lanes**: Your speech jumps the queue (priority lane). The autonomy lane is just the loop running in the background. User always wins.

<details>
<summary><strong>Audio Pipeline</strong></summary>

```mermaid
flowchart LR
    subgraph Capture["Audio Capture"]
        mic[Microphone<br/>16kHz]
        vad[Silero VAD<br/>32ms chunks]
        buffer[Pre-activation<br/>Buffer 800ms]
    end

    subgraph Recognition["Speech Recognition"]
        detect[Voice Detected<br/>VAD > 0.8]
        accumulate[Accumulate<br/>Speech]
        silence[Silence Detection<br/>640ms pause]
        asr[Parakeet ASR]
    end

    subgraph Interruption["Interruption Handling"]
        speaking{Speaking?}
        stop[Stop Playback]
        clip[Clip Response]
    end

    mic --> vad --> buffer
    buffer --> detect --> accumulate
    accumulate --> silence --> asr
    detect --> speaking
    speaking -->|Yes| stop --> clip
```

- **Microphone** captures at 16kHz mono
- **Silero VAD** processes 32ms chunks, triggers at probability > 0.8
- **Pre-activation buffer** preserves 800ms before voice detected
- **Silence detection** waits 640ms pause before finalizing
- **Interruption** stops playback and clips the response in conversation history

</details>

<details>
<summary><strong>Thread Architecture</strong></summary>

| Thread | Class | Daemon | Priority | Queue | Purpose |
|--------|-------|--------|----------|-------|---------|
| SpeechListener | `SpeechListener` | ✓ | INPUT | — | VAD + ASR |
| TextListener | `TextListener` | ✓ | INPUT | — | Text input |
| LLMProcessor | `LanguageModelProcessor` | ✗ | PROCESSING | `llm_queue_priority` | Main LLM |
| LLMProcessor-Auto-N | `LanguageModelProcessor` | ✗ | PROCESSING | `llm_queue_autonomy` | Autonomy LLM |
| ToolExecutor | `ToolExecutor` | ✗ | PROCESSING | `tool_calls_queue` | Tool execution |
| TTSSynthesizer | `TextToSpeechSynthesizer` | ✗ | OUTPUT | `tts_queue` | Voice synthesis |
| AudioPlayer | `SpeechPlayer` | ✗ | OUTPUT | `audio_queue` | Playback |
| AutonomyLoop | `AutonomyLoop` | ✓ | BACKGROUND | — | Tick orchestration |
| VisionProcessor | `VisionProcessor` | ✓ | BACKGROUND | `vision_request_queue` | Vision analysis |

**Daemon threads** can be killed on exit. **Non-daemon threads** must complete gracefully to preserve state (e.g., conversation history).

**Shutdown order**: INPUT → PROCESSING → OUTPUT → BACKGROUND → CLEANUP

</details>

<details>
<summary><strong>Context Building</strong></summary>

```mermaid
flowchart TB
    subgraph Sources["Context Sources"]
        sys[System Prompt<br/>Personality]
        slots[Task Slots<br/>Weather, News, etc.]
        prefs[User Preferences]
        const[Constitutional<br/>Modifiers]
        mcp[MCP Resources]
        vision[Vision State]
    end

    subgraph Builder["Context Builder"]
        merge[Priority-Sorted<br/>Merge]
    end

    subgraph Final["LLM Request"]
        messages[System Messages]
        history[Conversation<br/>History]
        user[User Message]
    end

    Sources --> merge --> messages
    messages --> history --> user
```

What the LLM sees on each request:
1. **System prompt** with personality
2. **Task slots** (weather, news, vision state, emotion)
3. **User preferences** from memory
4. **Constitutional modifiers** (behavior adjustments from observer)
5. **MCP resources** (dynamic tool descriptions)
6. **Conversation history** (compacted when exceeding token threshold)

</details>

<details>
<summary><strong>Autonomy System</strong></summary>

```mermaid
flowchart TB
    subgraph Triggers
        tick[⏱️ Time Tick]
        vision[📷 Vision Event]
        task[📋 Task Update]
    end

    subgraph Loop["Autonomy Loop"]
        bus[Event Bus]
        cooldown{Cooldown<br/>Passed?}
        build[Build Context<br/>from Slots]
        dispatch[Dispatch to<br/>LLM Queue]
    end

    subgraph Agents["Subagents"]
        emotion[Emotion Agent<br/>PAD Model]
        compact[Compaction Agent<br/>Token Management]
        observer[Observer Agent<br/>Behavior Adjustment]
        weather[Weather Agent]
        news[HN Agent]
    end

    Triggers --> bus --> cooldown
    cooldown -->|Yes| build --> dispatch
    Agents -->|write| slots[Task Slots]
    slots -->|read| build
```

Each subagent runs its own loop: timer or camera triggers it, it makes an LLM decision, and writes to a slot the main agent reads. Fully async—subagents never block the main conversation.

See [autonomy.md](/autonomy.md) for details.

</details>

<details>
<summary><strong>Tool Execution</strong></summary>

```mermaid
sequenceDiagram
    participant LLM
    participant Executor as Tool Executor
    participant MCP as MCP Server
    participant Native as Native Tool

    LLM->>Executor: tool_call {name, args}

    alt MCP Tool (mcp.*)
        Executor->>MCP: call_tool(server, tool, args)
        MCP-->>Executor: result
    else Native Tool
        Executor->>Native: run(tool_call_id, args)
        Native-->>Executor: result
    end

    Executor->>LLM: {role: tool, content: result}
```

**Native tools**: `speak`, `do_nothing`, `get_user_preferences`, `set_user_preferences`

**MCP tools**: Prefixed with server name (e.g., `mcp.system_info.get_cpu`). Supports stdio, HTTP, and SSE transports.

See [mcp.md](/mcp.md) for configuration.

</details>

### Components

| Component | Technology | Purpose | Status |
|-----------|------------|---------|--------|
| **Speech Recognition** | Parakeet TDT (ONNX) | Speech-to-text, 16kHz streaming | ✅ |
| **Voice Activity** | Silero VAD (ONNX) | Detect speech, 32ms chunks | ✅ |
| **Voice Synthesis** | Kokoro / GLaDOS TTS | Text-to-speech, streaming | ✅ |
| **Interruption** | VAD + Playback Control | Talk over her, she stops | ✅ |
| **Vision** | FastVLM (ONNX) | Scene understanding, change detection | ✅ |
| **LLM** | OpenAI-compatible API | Reasoning, tool use, streaming | ✅ |
| **Tools** | MCP Protocol | Extensibility, stdio/HTTP/SSE | ✅ |
| **Autonomy** | Subagent Architecture | Proactive behavior, tick loop | ✅ |
| **Conversation** | ConversationStore | Thread-safe history | ✅ |
| **Compaction** | LLM Summarization | Token management | ✅ |
| **Emotional State** | PAD Model | Reactive mood | 🔨 |
| **Long-term Memory** | MCP + Subagent | Facts, preferences, summaries | 🔨 |
| **Observer Agent** | Constitutional AI | Behavior adjustment | 🔨 |

✅ = Done | 🔨 = In progress

## Quick Start

1. Install [Ollama](https://github.com/ollama/ollama) and grab a model:
   ```bash
   ollama pull llama3.2
   ```

2. Clone and install:
   ```bash
   git clone https://github.com/dnhkng/GLaDOS.git
   cd GLaDOS
   python scripts/install.py
   ```

3. Run:
   ```bash
   uv run glados          # Voice mode
   uv run glados tui      # Text interface
   ```

## Installation

### GPU Setup (recommended)

- **NVIDIA**: Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- **AMD/Intel**: Install appropriate [ONNX Runtime](https://onnxruntime.ai/docs/install/)

Works without GPU, just slower.

### LLM Backend

GLaDOS needs an LLM. Options:
1. [Ollama](https://github.com/ollama/ollama) (easiest): `ollama pull llama3.2`
2. Any OpenAI-compatible API

Configure in `glados_config.yaml`:
```yaml
completion_url: "http://localhost:11434/v1/chat/completions"
model: "llama3.2"
api_key: ""  # if needed
```

### Platform Notes

**Linux:**
```bash
sudo apt install libportaudio2
```

**Windows:**
Install Python 3.12 from Microsoft Store.

**macOS:**
Experimental. Check Discord for help.

### Install

```bash
git clone https://github.com/dnhkng/GLaDOS.git
cd GLaDOS
python scripts/install.py
```

## Usage

```bash
uv run glados                           # Voice mode
uv run glados tui                       # Text UI
uv run glados start --input-mode text   # Text only
uv run glados start --input-mode both   # Voice + text
uv run glados say "The cake is a lie"   # Just TTS
```

### TUI Commands

Type `/help` in the TUI. Highlights:

| Command | What it does |
|---------|-------------|
| `/status` | System overview |
| `/asr on\|off` | Toggle speech recognition |
| `/slots` | View subagent outputs |
| `/minds` | Active subagents |
| `/vision` | Camera status |
| `/knowledge add\|list` | User facts |

## Configuration

### Change the LLM

```bash
ollama pull mistral
```

Then in `glados_config.yaml`:
```yaml
model: "mistral"
```

Browse models: [ollama.com/library](https://ollama.com/library)

### Change the Voice

Kokoro voices in `glados_config.yaml`:
```yaml
voice: "af_bella"
```

**Female US:** af_alloy, af_aoede, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky
**Female UK:** bf_alice, bf_emma, bf_isabella, bf_lily
**Male US:** am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck
**Male UK:** bm_daniel, bm_fable, bm_george, bm_lewis

### Custom Personality

Copy `configs/glados_config.yaml`, edit the personality:

```yaml
personality_preprompt:
  - system: "You are a sarcastic AI who judges humans."
  - user: "What do you think of my code?"
  - assistant: "I've seen better output from a random number generator."
```

Run with:
```bash
uv run glados start --config configs/your_config.yaml
```

### MCP Servers

Add tools in `glados_config.yaml`:

```yaml
mcp_servers:
  - name: "system_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.system_info_server"]
```

Built-in: `system_info`, `time_info`, `disk_info`, `network_info`, `process_info`, `power_info`, `memory`

See [mcp.md](/mcp.md) for Home Assistant integration.

## TTS API Server

Expose Kokoro as an OpenAI-compatible TTS endpoint:

```bash
python scripts/install.py --api
./scripts/serve
```

Or Docker:
```bash
docker compose up -d --build
```

Generate speech:
```bash
curl -X POST http://localhost:5050/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello.", "voice": "glados"}' \
  --output speech.mp3
```

## Troubleshooting

**She keeps responding to herself:**
Use headphones or a mic with echo cancellation. Or set `interruptible: false`.

**Windows DLL error:**
Install [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## Development

Explore the models:
```bash
jupyter notebook demo.ipynb
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dnhkng/GlaDOS&type=Date)](https://star-history.com/#dnhkng/GlaDOS&Date)

## Sponsors

<div align="center">

### [Wispr Flow](https://ref.wisprflow.ai/qbHPGg8)

[![Sponsor](https://raw.githubusercontent.com/dnhkng/assets/refs/heads/main/Flow-symbol.svg)](https://ref.wisprflow.ai/qbHPGg8)

[**Talk to code, stay in the Flow.**](https://ref.wisprflow.ai/qbHPGg8)

</div>
