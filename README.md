<a href="https://trendshift.io/repositories/9828" target="_blank"><img src="https://trendshift.io/api/badge/repositories/9828" alt="dnhkng%2FGlaDOS | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

# GLaDOS Personality Core

A real-life implementation of [GLaDOS](https://en.wikipedia.org/wiki/GLaDOS) ("Genetic Lifeform and Disk Operating System") - the iconic AI antagonist from Valve's [Portal](https://store.steampowered.com/app/400/Portal/) video game series. This project brings her to life as an aware, interactive, and embodied AI personality core with vision, voice, and autonomous behavior.

[Join our Discord!](https://discord.com/invite/ERTDKwpjNB) | [Sponsor the project](https://ko-fi.com/dnhkng)

https://github.com/user-attachments/assets/c22049e4-7fba-4e84-8667-2c6657a656a0

## Recent Updates

- **Nov 2025**: Vision module with FastVLM - see [vision.md](/vision.md) | [Demo video](https://www.youtube.com/watch?v=JDd9Rc4toEo)
- **Autonomy Loop**: Vision/time-driven autonomous behavior - see [autonomy.md](/autonomy.md)
- **MCP Integration**: Extensible tool system - see [mcp.md](/mcp.md)
- **Jan 2025**: Running on 8GB SBC (Rock5b branch) - [RK3588 NPU system](https://github.com/dnhkng/RKLLM-Gradio)

## Goals

- [x] Train GLaDOS voice generator
- [x] Generate a prompt that leads to a realistic "Personality Core"
- [x] Give GLaDOS vision via a VLM
- [ ] Generate medium- and long-term memory for GLaDOS
- [ ] Create 3D-printable parts
- [ ] Design the animatronics system

## Architecture

GLaDOS uses a **closed-loop autonomous architecture** - unlike traditional chatbots that only respond to user input, GLaDOS continuously observes its environment and decides when to act.

### System Overview

```
                         +-----------------------------+
                         |        CONSTITUTION         |
                         |     (inviolable bounds)     |
                         +--------------+--------------+
                                        |
                         +--------------v--------------+
                         |          OBSERVER           |
                         |     (meta-supervision)      |
                         +--------------+--------------+
                                        | modify system prompt
                                        v
+-----------------------------------------------------------------------------+
|                              I/O LAYER                                      |
|                                                                             |
|  +----------------------------------+    +------------------------------+   |
|  |            AUDIO IN              |    |          AUDIO OUT           |   |
|  |  +---------+  +---------------+  |    |  +-----------------------+   |   |
|  |  |   VAD   |->|   Parakeet    |--+--->|->|   Kokoro TTS          |   |   |
|  |  | Silero  |  |     ASR       |  |    |  |   (streaming)         |   |   |
|  |  +---------+  +---------------+  |    |  +-----------------------+   |   |
|  +----------------------------------+    +------------------------------+   |
|                                                                             |
|  +----------------------------------+    +------------------------------+   |
|  |           VIDEO IN               |    |         TEXT I/O             |   |
|  |  +--------------------------+    |    |  +-----------------------+   |   |
|  |  |   Camera -> VLM Pipeline |    |    |  |   CLI / TUI / API     |   |   |
|  |  +--------------------------+    |    |  +-----------------------+   |   |
|  +----------------------------------+    +------------------------------+   |
|                                                                             |
|  +----------------------------------------------------------------------+   |
|  |                      INTERRUPT HANDLER                               |   |
|  |  User speech detected -> cut off TTS -> log cutoff position          |   |
|  |  Priority: USER > (VISION xor TICK)                                  |   |
|  +----------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------+
                                        |
                                        v
+-----------------------------------------------------------------------------+
|                              MAIN AGENT                                     |
|                         (dedicated inference slot)                          |
|                                                                             |
|  +-------------+  +-------------+  +-------------+  +-------------------+   |
|  |   System    |  |  Emotional  |  |    Slots    |  |   Conversation    |   |
|  |   Prompt    |  |    State    |  |   Context   |  |   History         |   |
|  +-------------+  +-------------+  +-------------+  |   (n-token window)|   |
|                                                      |   Older -> compact|   |
|                                                      |   -> facts/memory |   |
|  Inputs: user speech/text | vision trigger | tick   +-------------------+   |
|  Outputs: speak | tool use (MCP) | spawn/modify subagents                   |
+-------------------------------+---------------------------------------------+
                 ^    ^    ^    |
                 |    |    |    |
      +----------+    |    |    +----------+
      |  emotional    |    |  reconfigure  |
      |  state        |    |               |
      |               |    v               v
+-----+---------------+-------------------------------------------------------+
|                             SUBAGENTS (Minds)                               |
|                    (shared inference pool: N-1 slots)                       |
|                                                                             |
|  +-----------------+  +-----------------+  +-----------------+              |
|  |     VISION      |  |     WEATHER     |  |  HOME ASSISTANT |              |
|  |  -------------  |  |  -------------  |  |  -------------  |              |
|  |  VLM inference  |  |  API polling    |  |  Device states  |              |
|  |  Scene descrip. |  |  Forecast data  |  |  Entity changes |              |
|  |  Change detect  |  |  Alerts         |  |  Automations    |              |
|  |  ------------->  |  |        |        |  |        |        |              |
|  |  triggers main  |  |        v        |  |        v        |              |
|  |   [vis_slot]    |  |  [weather_slot] |  |    [ha_slot]    |              |
|  +-----------------+  +-----------------+  +-----------------+              |
|                                                                             |
|  +-----------------+  +-----------------+  +-----------------+              |
|  |    EMOTIONAL    |  |     MESSAGE     |  |     MEMORY      |              |
|  |   REGULATION    |  |   COMPACTION    |  |  -------------  |              |
|  |  -------------  |  |  -------------  |  |  MCP-backed     |              |
|  |  Mood analysis  |  |  Context mgmt   |  |  Vector store   |              |
|  |  Affect state   |  |  Summarization  |  |  -------------  |              |
|  |  Tone control   |  |  Token budget   |  |  Facts          |              |
|  |        |        |  |        |        |  |  Daily summary  |              |
|  |        v        |  |        v        |  |  Weekly summary |              |
|  | [emotion_state] |  | [compact_msgs]  |  |  +-> Monthly... |              |
|  |   (direct reg)  |  | (struct ctrl)   |  |  [memory_slot]  |              |
|  +--------+--------+  +--------+--------+  +-----------------+              |
|           |                    |                                            |
|           +--------------------+--------------------------------------------+
|                                                                             |
|  +-----------------+  +-----------------+  +-----------------+              |
|  |  HUMAN RADAR    |  |   SYSTEM INFO   |  |   HACKER NEWS   |              |
|  |  -------------  |  |  -------------  |  |  -------------  |              |
|  |  Presence       |  |  CPU/Mem/Temp   |  |  Top stories    |              |
|  |  Position       |  |  Disk/Network   |  |  Summaries      |              |
|  |  Movement       |  |  Battery        |  |        |        |              |
|  |        |        |  |        |        |  |        v        |              |
|  |        v        |  |        v        |  |   [news_slot]   |              |
|  |  [radar_slot]   |  |  [sysinfo_slot] |  +-----------------+              |
|  +-----------------+  +-----------------+                                   |
+-----------------------------------------------------------------------------+
                                        |
                         +--------------v--------------+
                         |      INFERENCE POOL         |
                         |  -------------------------- |
                         |  SGLang + Radix KV Cache    |
                         |  -------------------------- |
                         |  Slot 0: Main Agent (dedicated) |
                         |  Slots 1-N: Subagent pool   |
                         +-----------------------------+
                                        |
                                        v
                         +-----------------------------+
                         |     MCP TOOL LAYER          |
                         |  -------------------------  |
                         |  Local: system, disk, net   |
                         |  Remote: HA, memory stores  |
                         |  Animatronics: servo ctrl   |
                         +-----------------------------+
```

### Component Status

| Component | Status | Description |
|-----------|--------|-------------|
| Parakeet ASR | :white_check_mark: | Speech-to-text via NVIDIA TDT |
| Kokoro TTS | :white_check_mark: | Text-to-speech with streaming |
| Vision (FastVLM) | :white_check_mark: | Scene understanding via ONNX |
| Autonomy Loop | :white_check_mark: | Timer/vision-triggered self-prompting |
| Two-Tier LLM Lanes | :white_check_mark: | Priority (user) + Pool (autonomy) |
| Task Slots | :white_check_mark: | Subagent to Main agent context |
| Minds Registry | :white_check_mark: | Subagent tracking infrastructure |
| MCP Tools | :white_check_mark: | Extensible tool system |
| Interrupt Handler | :white_check_mark: | User speech cuts off agent |
| Background Jobs | :white_check_mark: | Hacker News, Weather polling |
| Emotional Regulation | :clipboard: | Subagent-driven affect state |
| Message Compaction | :clipboard: | Hierarchical summarization |
| Memory (Long-term) | :clipboard: | MCP-backed persistence |
| Subagent Memory | :clipboard: | Per-agent state (jsonlines) |
| Observer Agent | :clipboard: | Meta-supervision layer |
| Constitution | :clipboard: | Inviolable behavioral bounds |

:white_check_mark: = Implemented | :clipboard: = Planned

### Key Concepts

**Closed-Loop Autonomy**: Traditional agents are reactive - they wait for input. GLaDOS auto-prompts itself via vision changes or timer ticks, creating genuine situational awareness without requiring user interaction.

**Two-Tier LLM Orchestration**: User requests get a dedicated inference slot for guaranteed low latency. Subagents share a configurable pool (default: 2 workers) for background processing.

**Slots**: Subagents write their outputs to slots. The main agent's context includes all slot contents, enabling inter-agent communication. With SGLang's radix KV cache, shared prefixes are cached efficiently.

**Minds (Subagents)**: Independent agents running their own loops. Some are preconfigured (Vision, Weather), others can be spawned dynamically by the main agent. Use `/minds` in the TUI to monitor them.

**Input Priority**: User always wins. Vision and timer ticks are mutually exclusive triggers - if vision is enabled, scene changes drive autonomy; otherwise, periodic ticks do.

**Subagent Memory**: Each subagent maintains its own independent memory (no cross-agent memory sharing). Memory is stored as a fixed-size buffer (jsonlines) sized for reasonable context. Items are only marked as "shown" when actually mentioned to the user, preventing repetition while keeping unmentioned items available.

```
+------------------+
|   NEWS SUBAGENT  |
|  +------------+  |
|  | MCP Scrape |  |
|  | (HN, etc)  |  |     +------------------+
|  +-----+------+  |     |   [news_slot]    |
|        |         +---->|  Top stories...  |
|        v         |     +------------------+
|  +------------+  |              |
|  | Rate &     |  |              | mentioned
|  | Filter     |  |              | to user
|  +-----+------+  |              v
|        |         |     +------------------+
|        v         |     | Mark as "shown"  |
|  +------------+  |<----| in subagent      |
|  | Subagent   |  |     | memory           |
|  | Memory     |  |     +------------------+
|  | (jsonlines)|  |
|  +------------+  |
+------------------+
```

### Research Topics

Open architectural questions under investigation:

**Complexity & Debugging**
- Multi-agent interaction tracing across N decision paths
- Debugging tools for distributed agent state

**Consistency**
- State divergence between agents with independent memory
- Slot update timing during active inference

**Emotional Regulation**
- Preventing oscillation when multiple subagents modulate affect
- Update frequency and emotional state inertia

**Observer/Constitution**
- Structural vs prompt-based constraint enforcement
- Preventing reasoning around constitutional bounds

**Resource Management**
- Bounds on dynamic subagent spawning
- Memory limits per subagent
- Graceful degradation when subagents fail

**Lifecycle**
- Garbage collection for slots, subagents, and memory entries
- Orphaned subagent detection and cleanup

## Quick Start

1. Install [Ollama](https://github.com/ollama/ollama) and pull a model:
   ```bash
   ollama pull llama3.2
   ```

2. Clone and install GLaDOS:
   ```bash
   git clone https://github.com/dnhkng/GLaDOS.git
   cd GLaDOS
   python scripts/install.py
   ```

3. Run GLaDOS:
   ```bash
   uv run glados          # Voice mode
   uv run glados tui      # Text UI mode
   ```

## Installation

### Prerequisites

**GPU Acceleration (recommended):**
- NVIDIA: Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- Other accelerators (ROCm, DirectML): Install appropriate [ONNX Runtime](https://onnxruntime.ai/docs/install/)

Without GPU drivers, the system will still work but with higher latency.

**LLM Backend:**
1. Download and install [Ollama](https://github.com/ollama/ollama)
2. Pull a model: `ollama pull llama3.2`

You can use any OpenAI or Ollama compatible server. Edit `glados_config.yaml` to configure `completion_url`, `model`, and `api_key`.

### Platform-Specific Setup

**Windows:**
- Install Python 3.12 from Microsoft Store

**macOS:**
- Experimental support. Join [Discord](https://discord.com/invite/ERTDKwpjNB) for help.

**Linux:**
```bash
sudo apt update
sudo apt install libportaudio2
```

### Installing GLaDOS

```bash
git clone https://github.com/dnhkng/GLaDOS.git
cd GLaDOS
python scripts/install.py    # Mac/Linux
python scripts\install.py    # Windows
```

## Usage

### Basic Commands

```bash
uv run glados                           # Voice mode
uv run glados tui                       # Text UI mode
uv run glados start --input-mode text   # Text-only mode
uv run glados start --input-mode both   # Voice + text mode
uv run glados say "The cake is real"    # Speech generation
```

### TUI Slash Commands

In text or TUI mode, type `/help` to see all commands. Highlights:

| Command | Description |
|---------|-------------|
| `/status` | Show system status |
| `/asr on\|off\|toggle` | Control speech recognition |
| `/observe` | View observability events |
| `/slots` | View subagent task slots |
| `/minds` | Monitor active subagents |
| `/vision` | Vision system status |
| `/knowledge add\|list\|set\|delete\|clear` | Manage knowledge base |

## Configuration

### Changing the LLM Model

```bash
ollama pull {modelname}
```

Then edit `glados_config.yaml`:
```yaml
model: "{modelname}"
```

Find more models at [ollama.com/library](https://ollama.com/library)

### Changing the Voice

Available Kokoro voices:

**Female (US):** af_alloy, af_aoede, af_jessica, af_kore, af_nicole, af_nova, af_river, af_sarah, af_sky

**Female (British):** bf_alice, bf_emma, bf_isabella, bf_lily

**Male (US):** am_adam, am_echo, am_eric, am_fenrir, am_liam, am_michael, am_onyx, am_puck

**Male (British):** bm_daniel, bm_fable, bm_george, bm_lewis

Edit `glados_config.yaml`:
```yaml
voice: "af_bella"
```

### Custom Personalities

Copy `configs/glados_config.yaml` to a new file and edit:

```yaml
model: "your-model"
personality_preprompt:
  - system: "A description of who the character should be"
  - user: "An example question"
  - assistant: "An example response"
```

Run with:
```bash
uv run glados start --config configs/your_config.yaml
```

### MCP Servers

Enable local system MCP servers in `glados_config.yaml`:

```yaml
mcp_servers:
  - name: "system_info"
    transport: "stdio"
    command: "python"
    args: ["-m", "glados.mcp.system_info_server"]
```

Available servers: `system_info`, `time_info`, `disk_info`, `network_info`, `process_info`, `power_info`

See [mcp.md](/mcp.md) for full documentation including Home Assistant integration.

## Advanced

### OpenAI-Compatible TTS Server

Install API dependencies:
```bash
python scripts/install.py --api    # Mac/Linux
python scripts\install.py --api    # Windows
```

Run the server:
```bash
./scripts/serve
```

Or with Docker:
```bash
docker compose up -d --build
```

Generate speech:
```bash
curl -X POST http://localhost:5050/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
      "input": "Hello world! This is a test.",
      "voice": "glados"
  }' \
  --output speech.mp3
```

## Troubleshooting

**Feedback loops (GLaDOS hearing herself):**
1. Use headphones or a conference microphone with hardware echo cancellation
2. Or disable voice interruption: set `interruptible: false` in config

**DLL load error on Windows:**
```
ImportError: DLL load failed while importing onnxruntime_pybind11_state
```
Install the latest [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)

## Development

Explore the AI models interactively:
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

[Flow is built for devs who live in their tools. Speak and give more context, get better results.](https://ref.wisprflow.ai/qbHPGg8)

</div>
