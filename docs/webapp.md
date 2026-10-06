# Webapp Console

An in-process "mission control" for the running GLaDOS engine. It streams the
live parallel state — the two autonomy lanes, subagent contexts, tool state,
PAD emotion, audio/MCP health — to a browser over HTTP + Server-Sent Events.
No separate UI service to run. Tool argument validation uses jsonschema.

## Decoupled launcher (key design)

The webapp console is **not** part of the core engine. It is started by a
dedicated CLI command — `glados webapp` — that mirrors how `glados tui` works:
it loads the config, builds a `Glados` engine, starts the in-process
`WebappServer` on its own port, then runs the engine loop and shuts the server
down when the loop exits. The engine itself holds no webapp knowledge.

The observable state (`ObservabilityBus`, `MindRegistry`, `TaskSlotStore`,
subagent memory, interaction/emotion state) only exists inside the running
`Glados` object, so the console server runs in the same process and reads those
objects directly — exactly the pattern the WebSocket audio backend uses. This
keeps it dependency-light and side-effect free for every engine entry point
(`start`, `tui`, `say`).

The webapp and the TUI are **mutually exclusive UI options** — you run either
`glados webapp` or `glados tui`, never both; the core engine stays agnostic to
which one is attached.

## Running it

**Search Core** handles requested internet research through the Exa MCP tool.
It searches, uses a background inference to identify useful quotations and gaps,
and follows gaps with focused queries. It stops when the evidence is sufficient,
or returns partial findings when a limit or service failure prevents completion.
Selected quotations must occur in the returned excerpts; citations retain the
returned URLs. This validates attribution, not the truth of a source's claims.

Pending background searches receive only a fixed one-sentence acknowledgement:
"I'm checking that now." or "Your search is queued." This uses no reply inference
and includes no camera observations or other context. Completed reports still
receive a normal answer. For a general news briefing, two or three timely,
distinct headlines are sufficient; seeking additional headlines is not itself
an unresolved gap. Hacker News listings are attributed as aggregator listings,
with the returned URL, rather than presented as independent verification.

Search Core receives the live host clock and resolves ordinary relative days
into absolute dates before searching. The date stays pinned across follow-up
searches, including when a request spans midnight. Weather findings must include
a matching calendar label in the quoted forecast row or paragraph; a "tomorrow"
URL, publication date or weekday alone is insufficient. Undated and wrong-day
forecasts are excluded from the final report, including fallback excerpts. If
dated evidence is unavailable after follow-up searches, Central reports that
verification failed for the requested day. It does not ask the user for the
current date, blend hourly readings into daily highs/lows, or blame the request.

Defaults are three searches, two results per query, six sources, a 60-second
research budget and a 3,500-character report. Each network/review request has a
15-second timeout. Reviews share the existing background inference slots; no
dedicated model or extra GPU slot is allocated. It performs no periodic browsing.

**Facility Settings → Internet search sources** provides editable Weather, News,
Reddit communities and General lists. Enter one domain or URL path per line,
up to eight per category. Weather defaults to `dwd.de` and `meteoblue.com`; News
to `reuters.com` and `bbc.com/news`. Reddit and General start empty. A subreddit
can be entered as `reddit.com/r/LocalLLaMA`. Clear a list to use no preference.
For news requests, Search Core reads the configured News pages directly over
HTTP before using a search engine. A domain means its home page; a path selects
that page. This reads Hacker News and HuggingNews's visible HTML headlines,
links and update labels, without executing scripts. Each download is capped at
500 KB, each evidence excerpt at 3,500 characters, with a bounded page-reading
budget inside the existing research deadline. Blocked, unreadable or insufficient
pages fall back to web search. The fetch timestamp is not an article publication
date. Other categories favor the relevant category and General sources in their
first search. Preferred sites never bypass date or quotation checks, and
community posts remain community claims. Saving applies to the next research
request and persists in `data/search_settings.yaml` across restarts.

All saved application settings use human-editable YAML. The main profile is the
YAML passed to `--config`; console overrides are saved atomically in these files:

| File | Settings |
| --- | --- |
| `data/search_settings.yaml` | Weather, News, Reddit and General sources |
| `data/vision_settings.yaml` | Observation timing range |
| `data/decision_lists.yaml` | Routing lists, options and thresholds |
| `data/operator_settings.yaml` | Response instructions |
| `data/preferences.yaml` | Saved user preferences |

File edits take effect after restarting GLaDOS; console edits also update the
running settings. On first use, a corresponding legacy JSON settings file is
read and migrated to YAML. YAML takes precedence thereafter, and the old JSON
file is retained unchanged as a backup. These local overrides are ignored by Git.

For example, the News list can be edited directly:

```yaml
weather:
  - dwd.de
  - meteoblue.com
news:
  - news.ycombinator.com
  - huggingnews.com
reddit: []
general: []
```

With Autonomy enabled, existing background tasks deliver completed research to
Central Core. With Autonomy disabled, Central receives the completed report as
the tool result. The latest report also supplies bounded context through Search
Core's slot, cleared for a new user request. Research can be suspended from its
Cores inspector, which shows progress, sources and cited passages. Only the latest
report is retained in memory; no separate research archive or log file is created.

```yaml
Glados:
  search:
    enabled: true
    max_rounds: 3
    results_per_query: 2
    deadline_s: 60
    max_report_chars: 3500
    preferred_sources:
      weather: [dwd.de, meteoblue.com]
      news: [reuters.com, bbc.com/news]
      reddit: []
      general: []
```

**Health Core** appears in Cores and as a dedicated Neural Buffer context source.
It samples CPU load, RAM, local disk space, uptime, OS/kernel, temperatures and
NVIDIA GPU readings, plus model/MCP connectivity, inference queues and microphone
capture health. Unsupported readings are unavailable; cached readings include
their timestamp and age. Paused or old readings are explicitly stale.

Stats are sampled with ordinary code; rolling commentary uses the shared
background inference pool. Sampling continues while a comment is queued, and
only one comment can be in flight. A comment is generated every 60 seconds by
default, yielding to interactive work. Questions read the existing context
instead of starting another health probe or commentary request. Health conditions
emit one alert when they appear and a recovery event when they clear; routine
samples do not produce Info messages or unsolicited speech. Autonomous speech
still follows the existing autonomy setting. No separate statistics history is
written to disk.

```yaml
Glados:
  health:
    enabled: true
    interval_s: 10
    max_age_s: 30
    summary_enabled: true
    summary_interval_s: 60
    summary_max_tokens: 128
    # Optional: monitor known log files and their numbered rotating backups.
    log_paths: []
```

The capture runner's own log is detected automatically on Linux. Explicit
`log_paths` can add the model log or other known files; Health does not scan
directories for arbitrary logs. Alerts observe conditions and do not perform
repairs. Current measurements take precedence over an older model comment.

The **Test Chamber** tab browses saved facts, saved summaries and the Memory Core's
existing compacted conversation notes. It shows the memories currently recalled
for the conversation separately. Search, fact/summary filtering and paging are
read-only; browsing never changes the active recall topic. Saved text is escaped
for display and excerpts are labelled. Memory responses are same-origin and
not cached by the browser.

Memory Core clears the previous recall before routing and starts a lookup once
the user turn is accepted. Retrieval runs alongside response generation; Central
can use a result that arrives before its context is built. Later relevant results
publish an important slot update so Autonomy can compare them with the actual
answer and decide whether a useful follow-up remains. New turns supersede old
lookups, and at most one latest query waits behind an active lookup. Topic word overlap is
ranked by specificity, then recency and importance. Relevant low-importance facts
can be recalled; unrelated high-importance facts are not injected automatically.
Short references such as “what about it?” can reuse the previous user topic.
Recall currently uses local text matching, without an embedding model or another
LLM call. Voice recall uses the optional transcript; raw audio without a transcript
has no text query. Pausing Memory Core clears its recalled context.

The bounded recall is published in the Memory Core slot's `context` field and
included by the shared slots context source. Maintenance statistics stay in the
core's report. Recalled facts retain their source and saved date, are treated as
quoted data, and defer to current requests and later corrections. Each refresh
replaces the prior selection, including when no facts match. No additional archive
or recall log is written.

```yaml
Glados:
  autonomy:
    tokens:
      recall:
        enabled: true
        memory_dir: ~/.glados/memory
        max_facts: 6
        max_chars: 2400
        include_summaries: true
```

Recall can run with compaction disabled (`tokens.enabled: false`). Reads of the
existing facts/summaries files are cached until they change and bounded to the
newest 8 MiB per file, with at most 2,048 indexed memories overall. The memory
browser reports when that bound applies. Context is capped independently, and
the existing compacted conversation store is reused without duplicating it on disk.

The **Neural Buffer** tab inspects Central Core or Autonomy Core context in model
message order. **Live preview** uses the same source assembly as inference and
refreshes every two seconds while visible; **Last submitted request** retains
the actual request messages and offered tools, including routing instructions,
tool continuations and current input. Expand sources to inspect the system
prompt, session preferences, PAD tone, task/job results, retrieved memory, MCP
context, vision observation, compacted notes and chat history. Search, pause
updates or download the inspection as JSON. Binary media is represented by
placeholders; it is not retained in the inspection. Viewing context performs no
inference and does not modify the conversation.

Each section header shows its actual message-number range and explains its purpose.
The order is reusable instructions, completed history, live state, then current
input and tool continuations. **Show all messages** expands the payload without
regrouping it by core. **Session preferences** supplies language, length and style;
**Emotion Core** supplies the live PAD values and current tone. **Tool and task
rules** replaces the long Facility instructions block with concise execution rules.
Dedicated emotion, vision and compaction statistics are omitted from the task list;
their full reports remain in Cores. Language defaults live in session preferences,
and animation syntax lives in the separate speech-format instructions.

For ordinary replies, tool definitions are a separate API field, not extra numbered messages. In a live
preview, the tool catalogue shows available capabilities and is explicitly labelled
as a preview; routing chooses the subset for the real request. Last submitted
request shows exactly the offered definitions. Connected service data, when present,
comes only from configured MCP resources and is distinct from that catalogue.
Autonomy has its own reviewer policy and JSON decision protocol, existing chat
summaries/recent turns, and all core/task slot evidence. It offers no tools. A validated
no-action decision ends the check; a prompt passes evidence and an internal instruction
to Central Core, which composes the notification.

**Facility Settings → Facility Sensors & Audio** lists the host's webcams,
microphones and speakers. Selections apply to the running backends for the
current session. Linux cameras are enumerated with V4L2 capability queries, so
metadata-only nodes are excluded. Audio lists are filtered by input/output
capability, include the system default and support native-rate stereo
microphones by converting to 16 kHz mono VAD chunks using existing numpy code.
An input opening failure restores the previous microphone. Camera changes
discard prior frames and visual comparisons; camera OFF and microphone mute
remain unchanged. Local audio selectors are unavailable with the WebSocket
audio backend. Permanent defaults can be set with `vision.camera_spec` and
`audio_io_options.input_device` / `output_device` in the YAML configuration.

On Linux, **System default** audio uses the desktop PulseAudio/PipeWire route
when available, so hardware changes do not pin capture to an obsolete ALSA
handle. The speech listener checks capture health once a second and reopens a
stalled or inactive microphone, with three-second retry backoff if the device
is absent. It never restarts an intentionally stopped capture. Playback uses
the driver's higher-latency buffer to tolerate scheduling jitter, and device
inspection reports capture health/recovery counts and playback underruns.
Microphone buffering is capped at about one second. A capture overflow discards
the incomplete turn rather than replaying old speech later. Speech onset needs
96 ms of sustained VAD, or 160 ms while a reply is playing. Each new user turn
invalidates the previous turn's pending inference, tools, synthesis and playback;
setting the processing flag again cannot revive an interrupted reply.

The front-page optic feed shows **MOTION** at bottom left, measured from actual
frame differences in camera coordinates. **TEST SUBJECT** appears at top right
only while a fresh face target is present. Face presence and animated panning
do not themselves increase the motion reading.

The console is **off by default**. Enable it, then start it with the webapp
launcher:

1) YAML config:

   ```yaml
   webapp:
     enabled: true
     host: 127.0.0.1
     port: 8050
   ```

2) Environment variables (no config edit):

   ```bash
   GLADOS_WEBAPP_ENABLED=1 GLADOS_WEBAPP_PORT=8050 uv run glados webapp
   ```

Both can be combined with `--config`, `--input-mode`, `--tts-enabled`/
`--tts-disabled`, and `--asr-muted`/`--asr-unmuted`:

```bash
uv run glados webapp --config ./configs/glados_webapp_config.yaml
```

Then open `http://127.0.0.1:8050/`.

> **Demo mode.** A directly-opened static file can use simulated data for
> styling. An HTTP console with an unavailable engine shows a disconnected
> state and retries; it never substitutes simulated activity.

## Operator controls and shared tasks

### Cores and facility controls

The interface uses Aperture Science terminology. **Cores** replaces the separate
Minds and Slots pages. Its cards represent the Central Core (GLaDOS), Emotion Core,
Vision Core, Memory Core (recall and conversation compaction), Health Core,
Search Core, Routing Core and Autonomy Core.
Each card shows its function, latest output and live processing state. Working and
Queued come from admitted/waiting inference requests; a running thread alone is
shown as Standby. Suspension stops future scheduled work; an admitted run may finish.
Inspect a core to read its report and memory, suspend/resume it, or run it once.

The Cores header shows shared processing capacity. Expand **Shared processing**
to inspect active channels, owners, queued work, the latest routing decision and
facility command execution. Channel capacity remains shared among cores; cores
are not permanently assigned to backend slots. Existing API IDs and routes remain
compatible, including `/api/minds` and `/api/slots`.

- **Test Assignments** contains saved tasks and results. Core reports live in their
  inspectors instead of the assignment board.
- **Facility Settings** manages persistent decision lists and previews text or audio routing.
  Its **Vision Core** setting controls a motion-biased random image analysis delay between two integer bounds
  (1–60 seconds, default 2–5). Changes apply to the running schedule and persist in
  `data/vision_settings.yaml`, overriding the profile range on restart. CPU face
  tracking keeps its own cadence; the avatar blink follows actual Vision inference.
- **Facility Tools** lists the actual callable capabilities exposed to the reply agent,
  including connected MCP tools, descriptions, and input schemas. Command tools
  have direct local tests. Other tools are invoked through a request to GLaDOS.
- **Diagnostics** contains engine threads, device health, and expandable inference
  diagnostics. The separate Neural Core page has been removed because it
  duplicated this infrastructure information.

Each user-facing response receives a fresh host-clock reading near the current
input, after routing, Emotion and inference admission. The context explicitly
labels local time, date, weekday, timezone, UTC offset and the capture timestamp.
It is authoritative for that response and is never saved as conversation history.
Local clock questions use the ordinary conversational reply route, including
native audio with transcripts disabled. There is no Clock choice in the routing
tree. Speculative replies capture the clock when their inference starts.

`get_time` is reserved for another explicitly named IANA timezone and requires its
`timezone` argument. The redundant local-clock command and console clock RPC have
been removed. Saved local `get_time`, command-clock and context-clock choices
are removed during migration; named timezone choices remain tools under Local
system. Other custom choices keep their IDs and descriptions.

For routed speech, the conversation keeps an explicitly labelled interpretation
of the chosen action. Tool results carry that action into the reply request, so
the assistant can answer without requiring a transcript or retaining raw audio.

`run_safe_command` takes one `task`: `uptime`, `disk_usage`, `memory_usage`,
`system_info`, or `cpu_load`. Each maps to a fixed Linux executable and argument tuple.
CPU load reads `/proc/loadavg`: the first three numbers are 1, 5 and 15 minute
load averages, not CPU utilization percentages.
The shared command slot runs one subprocess at a time, with no shell, no stdin,
a clean environment, a three-second timeout, and output capped at 4096 characters
per stream. It rejects custom commands, extra arguments, and requests while busy.
Missing commands return an error on unsupported hosts. Execution uses CPU capacity,
not a model slot; interpreting the request and speaking the result still use inference.
Run commands from Cores → Shared processing or the tool inspector; `POST /api/tools/command` accepts
`{"task":"uptime"}`. Live state includes `command_slot`, with active work and the
last 20 results; the console shows the most recent five. Settings decision options
can bind the tool to a fixed task using its enum input.
New decision settings include fixed choices for uptime, disk, memory, CPU load and system
information when the command tool is available. Existing saved lists are preserved;
add those choices in Settings as needed. Explicit uptime choices prevent a generic
time-related request from being confused with the clock.

Local E4B validation on 2026-10-05 covered two synthesized spoken time questions
and two spoken uptime requests with transcripts disabled, plus a typed command request,
including previous history
that wrongly demanded a transcript. Both clock replies matched the tool reading;
all three command requests selected the fixed uptime option and summarized its output.
The generic five-option list had confused spoken uptime with clock time; the explicit
command choices passed at the unchanged 0.8 preference threshold and 0.2 margin.
Raw results are in
`docs/benchmarks/command-time-2026-10-05.json`. These timings stop at generated speech
text; they do not include speech synthesis or playback.

Enrichment Center sends typed messages to the same reply agent as microphone
input. Replies appear on the front page even when voice output is muted.
Brainstem controls microphone capture, voice output, and camera observations.
When VAD detects microphone speech, the eye shows a cyan attention ring,
opens its aperture slightly, and holds an attentive pose. The cue fades after
speech ends and wakes a sleeping eye. Microphone activity does not drive the
speaking animation. A compact `audio` SSE event delivers VAD and RMS updates
at up to 30 Hz without repainting the dashboard; timestamps keep slower state
snapshots from undoing newer speech activity. Reduced-motion mode retains the
static attention ring. GLaDOS's own playback takes precedence over listening.
Settings → Voice input controls optional direct-audio transcripts. Changes are
acknowledged by the engine before the UI changes state; controls are disabled
while disconnected. In transcribed-audio profiles, transcription is required.

Brainstem displays the current response instructions inline. Edit Response
Instructions opens the response instructions included in the next model
request. Use Defaults fills the editor; Save applies it and persists it in YAML. The reply agent also
receives instructions explaining task ownership, reports, completion criteria,
tool failures, and the distinction between avatar expressions and PAD emotion.
Context and emotion inspectors expose the actual engine state. Unavailable PAD
gauges are labeled Off rather than displaying invented readings.

Emotional regulation follows `autonomy.emotion.enabled` (default true)
independently of `autonomy.enabled` and `autonomy.jobs.enabled`. It uses the
configured model and thinking settings, updating every
`autonomy.emotion.tick_interval_s` seconds (default 5), even without new events.
Accepted voice or text input updates affect immediately before the reply and
restarts the five-second timer after the reaction finishes. Queued vision and tool
events are processed at the next scheduled update. Elapsed-time exponential
decay removes 95% of a PAD deviation in six minutes (`decay_settle_s: 360`),
returning all axes toward zero. This does not enable
background jobs or load another model. PAD represents GLaDOS's modeled persona,
not a measurement of the user's emotions.

The shared task board works even when autonomy is disabled. Create or edit a
task, then choose **Ask GLaDOS to work on this** to submit it to the reply agent.
The `manage_slot` tool saves the result and status under the same task ID; the
inspector shows the full report and refreshes when it changes. Tasks can be
marked done or reopened manually. Agent-owned slots remain read-only.

Tasks and response instructions are in memory for this engine session and reset
on restart. Creating an open task does not start a background job or schedule a
reminder. Cores and inference lanes show registered components and actual worker
capacity, including emotion requests when autonomy jobs are disabled.

## GLaDOS avatar

Enrichment Center includes the aperture rig from `GLaDOS Concept Stills.html`,
served locally without external fonts or animation dependencies. It retains the
original blade mechanics, mood blending, blinks, gaze, and speech motion.

- Spoken emotion directions select the expression during playback. PAD emotion
  provides the fallback expression; this never modifies engine emotion.
- User-facing inference selects the processing expression while she is not speaking
  or listening. Microphone VAD adds a cyan attention ring, a wider aperture,
  and an attentive pose that fades after speech ends.
- Playback events enable an illustrative speech pulse. This is not waveform
  synchronization: the browser receives controls, not a WAV or TTS amplitude.
- A disconnected stream closes the eye instead of continuing stale activity.
- YuNet tracks faces on the CPU independently of E4B scene updates. Gaze follows
  faces and localized movement automatically when Camera is ON and the mouse
  pointer when it is OFF. The optic feed's existing full-image motion map finds
  movement outside the current crop. A wave draws a brief 0.9-second glance at
  the moving region, then the eye returns to the same face. A three-second
  cooldown gives eye contact priority over continuing movement. Multiple faces
  still alternate, and separate moving regions remain separate targets. Motion
  inside a face box, isolated noisy pixels, and broad exposure/camera changes
  do not select new targets. This runs locally in the browser without another
  camera stream or model inference. The eye and image reticle share the selected
  target; the feed labels movement as **MOTION**.
  Small camera SSE events deliver new face positions at up to 30 Hz, without
  repainting the dashboard or updating E4B captions. Camera gaze takes priority
  over idle glances. A scan alternates approximate eye positions with briefer
  mouth fixations around the moving face box; its offsets scale with face size
  and diminish during quick head movement. Nearness reaches full zoom when
  either face dimension fills two-thirds of the image, smoothly enlarging the
  entire eye up to 1.6×. Camera X is reversed for the eye to follow the viewer across
  the screen; the preview remains in camera coordinates. Optional Haar/E4B
  backends are available in vision configuration.
  Five seconds of confirmed absence puts the idle eye to sleep; localized movement,
  a face, speech, a user
  request, or console interaction wakes it. Paused or stale cameras cannot trigger
  sleep. This changes the animation only, without stopping audio or inference.
  Uncertain or malformed observations cannot trigger sleep.
- Animation runs automatically. Reduced-motion preferences show static expressions. Animation stops in other console views
  and hidden tabs, and is capped at 30 frames per second when visible.

The live console uses engine playback controls and camera presence instead of
the concept page's voice-file playback and face-tracking stand-ins.

### Spoken emotion directions

The user-facing LLM receives a system instruction allowing interleaved speech
and silent directions, for example:

```text
[emotion:smug]Excellent work. [emotion:disappointed]For a human.
```

Supported expressions: `neutral`, `quizzical`, `angry glare`, `suspicious`,
`smug`, `surprised`, `bored`, and `disappointed`. Tags can span streaming chunks.
The parser removes them before TTS and carries the expression alongside each
speech segment. Unknown tags retain the previous expression; unfinished tags
are discarded at end of response. Untagged responses continue to work.

Directions reset for each response. An expression becomes active when its
audio segment starts playing, then clears on completion, interruption, or error.
The TTS voice itself is unchanged. There is no browser WAV decoding or second
audio stream, and autonomous internal reasoning gets no direction prompt.

## Endpoints

| Method | Path                  | Purpose |
| ------ | --------------------- | ------- |
| GET    | `/`                   | Static console (`static/index.html`). |
| GET    | `/api/vision/live`    | Live MJPEG webcam stream with CPU face overlay, up to 30 fps |
| GET    | `/api/vision/frame`   | Selected observation JPEG, kept in memory; same-origin access. |
| GET    | `/api/snapshot`       | Aggregate JSON snapshot (minds, agents, slots, lanes, audio, emotion, MCP, interaction, vision). |
| GET    | `/api/state`          | Lightweight state JSON for the live gauges. |
| GET    | `/api/stream`         | SSE stream (see contract below). |
| GET    | `/api/minds`          | Registered mind statuses. |
| GET    | `/api/minds/{id}`     | Single mind status. |
| GET    | `/api/minds/{id}/memory` | That agent's private jsonlines memory entries. |
| GET    | `/api/slots`          | Task slots (summary fields). |
| GET    | `/api/slots/{id}`     | Full slot including the on-demand report. |
| GET    | `/api/agents`         | Subagent statuses (`agent_id, title, running, tick_count, last_tick`). |
| POST   | `/api/command`        | `{"command": "/agents"}` → run an engine command (mirrors the TUI palette). |
| POST   | `/api/input`          | `{"text": "..."}` → submit a user message to the reply agent. |
| POST   | `/api/control`        | `{"action": "microphone", "enabled": false}`; also `voice` and native-audio `transcripts`. |
| POST   | `/api/instructions`   | `{"instructions": "..."}` → save response instructions (up to 4,000 characters). |
| POST   | `/api/vision/settings` | Save `interval_min_s` and `interval_max_s` (integers 1–60, minimum ≤ maximum); update the running motion-aware Vision schedule. |
| POST   | `/api/slots`          | Create a task with `title` and `summary`; update using `slot_id`, with optional `status` and `report`. |
| POST   | `/api/minds/control`  | `{"agent_id": "emotion", "action": "pause"}`; actions are `pause`, `resume`, or `run`. |

Snapshots include `agent_minds` for the operational agents and `tools` for their
callable capabilities. The legacy `minds` field contains engine registry entries
used by the Vitals diagnostics.

Mutation endpoints require JSON, reject cross-origin browser requests, and
validate body size and field values. This remains a local console without an
authentication layer; keep the default loopback binding.

## Capability routing

Settings shows the live routing tree as an expandable diagram. Teal nodes and
links trace the latest request or test preview, with the accepted step scores;
a yellow node marks an uncertain step. Use **Overview**, **Expand all**, or
**Show latest route** to change the view. The full descriptions remain below it.
**Download Mermaid** exports the actual tree as a `.mmd` flowchart, including
the current MCP servers and tools; its source is also available to copy.

Routing defaults to a short hierarchy. The first one-token choice selects
Vision, Memory, MCP services, Local system, Tasks, Other tools, or a conversational
outcome. Only available capabilities appear; internal speech and no-op tools are
excluded. A branch with just one variable-argument tool goes directly to argument
filling. Memory and Local system select among their tools and fixed bindings.
Local time/date questions use ordinary conversation and the fresh host-clock
context. Questions about another named timezone use Local system. MCP requests select
a connected server, then choose a tool only if needed. The connected MCP Slow
Clap replaces the local entry unless an operator has an explicit local playback
binding; local playback remains available when disconnected. Large catalogs
split into groups of at most 19 choices per step.
Letters restart at A within each step; saved option IDs remain unchanged.

The MCP manager injects registered tool names and descriptions into the routing
prompt on every request. It uses the allowed live registry, without fetching
context resources or exposing connection credentials. Optional MCP config
`description` describes the server's purpose; `routing_category: memory` places
a memory service under Memory. A server named `memory` is recognised automatically.

Each step scores one token with the existing complete option probabilities and
must pass the list's threshold and margin. The whole route shares one timeout.
An uncertain step or an explicit “No matching choice” stops the pipeline and
uses the configured fallback, without authorizing any tools. Missing arguments
alone are not a mismatch; the assistant fills them for the selected tool. These
scores express preference among the options, not calibrated certainty. Ordinary
conversation and single-tool areas such as Vision take
one step; branches with multiple choices normally take two. Single-tool MCP
servers take two steps, and servers with multiple tools take three. With
transcripts disabled, native audio is carried unchanged through all steps.
When optional transcripts are enabled, E4B transcribes once and routing, Emotion
and the response reuse that text. If transcription fails or is empty, the original
audio remains available.

Fixed bindings retain their versioned execution permits. For variable arguments,
the assistant receives the original request and only the tools selected by the
route. It fills arguments from that request. Pending generated calls are checked
again against the active list, settings revision, selected scope and current
availability before execution. Editing settings revokes an old scoped call.
Parallel private drafts may begin during the first routing step; only a final
reply outcome releases the draft, and tool routes discard it.

Settings shows the live routing tree and groups fixed choices by capability.
Routing is enabled automatically on compatible llama.cpp backends; other backends
use the reply model's tool planning. There are no routing or parallel-draft
switches. Selecting an active list saves immediately. Deleting or disabling the
active list selects another enabled list, restoring a default if none remain.
Old saved routing switches are migrated without losing custom lists.
Operators can edit an option's capability assignment, instructions, thresholds,
arguments and enabled state. Existing lists adopt the hierarchy without losing
saved IDs or bindings; the list editor also offers the previous flat structure.
Preview runs all needed steps and displays each probability table without
executing tools or producing speech.

## SSE contract — `/api/stream`

Each connection first replays the last 100 events from the bus, then receives:

- `obs` events — the same shape the TUI `ObsScreen` renders:

  ```json
  {"timestamp": 1750000000.0, "source": "autonomy", "kind": "slot.update",
   "level": "info", "message": "weather brief -> done", "meta": {"slot": "s_weather"}}
  ```

  Real `source`/`kind` combos include `llm.request`, `llm.queue`,
  `llm.tool_calls`, `autonomy.dispatch`, `autonomy.slot.update`,
  `subagent.start/stop`, `tool.start/finish/error/timeout`, `tts.*`,
  `mcp.*`, `vision.observation`, `text.user_input`.

- `state` events — every ~0.5 s, mirroring `/api/state`, so gauges and lane
  chips stay live even during continuous observability traffic. Queue depths
  and active inference counts are separate for each lane. Audio RMS is linear;
  the browser converts it to decibels for display.
- `snapshot` events — immediately after connection and every ~2 s, mirroring
  `/api/snapshot`, to refresh minds, agents, slots, and vision without reloading.
- `audio` events — `{rms, vad_active, updated_at}` sampled at up to 30 Hz when
  audio changes. These drive the avatar's listening cue without repainting
  the console. `updated_at` prevents stale snapshots from overriding live VAD.
- `camera` events — `{paused, camera, inference_sequence, inference_active}`
  with new face positions at up to 30 Hz; they update only the avatar. Slower
  snapshots cannot overwrite newer face coordinates. Camera OFF/disconnection
  clears tracking normally. Vision inference sequences trigger a shared blink.
- `performance` events — immediate playback controls such as
  `{"revision": 5, "active": true, "emotion": "smug", "updated_at": 1750000000.0}`.
  Completion sends `active: false, emotion: null`. Snapshots and state also
  contain the current `performance`, allowing late joiners to recover. The
  browser ignores older revisions; historical `obs` events never replay speech.

The connection badge distinguishes live telemetry, a disconnected stream that
is retrying, and demo data.

### Multi-consumer (the important bit)

The TUI's `ObservabilityScreen` consumes the bus via `drain()`, which is a
*single-consumer* FIFO. The webapp never calls `drain()`. Instead every SSE
connection registers its own private `subscribe()` queue on
`ObservabilityBus`, so multiple browsers each get their own copy and never
steal events from the TUI or from each other. This is fully backward-compatible:
`drain()` and `snapshot()` keep their existing behavior.

## Failure behavior

Because the webapp is the point of the `glados webapp` launcher, a disabled
console or a bind failure (port in use) is **fatal for that command**: the
launcher logs an error and exits rather than running the engine without the
console. This does not affect other entry points — `tui`, `start`, and `say`
are completely independent of the webapp and never bind a port.

- Request exceptions return JSON error bodies and never crash a worker thread.
- Client disconnects free their subscription.

## Lifecycle

The `glados webapp` command orchestrates the whole lifecycle:

1. Load `GladosConfig.from_yaml` and apply CLI overrides.
2. Refuse to start if `webapp.enabled` is false.
3. Build `Glados.from_config`, then `WebappServer(engine, host, port).start()`.
4. Run `engine.run()`; on any exit (`try/finally`), shut the server down.

The core engine stays decoupled and side-effect free. This mirrors the TUI's
launcher pattern: the same `GLADOS_WEBAPP_*` environment variables live on
`GladosConfig.webapp`, so the launcher reads one merged config.
`GladosConfig.webapp` stays a field on the shared config model, but the engine
never reads it at runtime — the decoupling is at the launcher boundary.

## Development

### Logging limits

Info shows user input, replies, routing/tool activity, controls and lifecycle
events. Continuous camera captions, private core state, queue timing and speech
pipeline diagnostics use Debug. The home stream omits Debug; the Observation
Log's `debg` or `all` filter retains it. Unchanged slot refreshes emit no event.
Warnings and errors remain visible.

Observability stays in memory: 500 history/drain entries and 100 entries per
subscriber. Browser history holds 400 events, and the TUI observation display
holds 500 lines. The application does not configure a file logging sink.

Shell output redirection (`>file.log`) bypasses these limits. For persistent
process logs, use the bounded capture launcher from the repository root:

```bash
python3 scripts/run_with_logs.py --log-file /tmp/glados-webapp.log -- .venv/bin/glados webapp --config configs/glados_webapp_config.yaml
python3 scripts/run_with_logs.py --log-file /tmp/glados-model.log -- bash scripts/run_llamacpp.sh
```

Each command keeps one 5 MiB file and two backups (about 15 MiB total).
Capture reads bounded chunks, including output without newlines, and forwards
SIGINT/SIGTERM to the command's process group. It preserves the command's exit
status. The model launcher defaults to warnings/errors; append
`--log-verbosity 3` for model request timing diagnostics when needed.
Docker Compose also limits container logs to three 5 MB files.

- Console page: `src/glados/webapp/static/index.html` — self-contained console
  page (Aperture/GLaDOS theme, no build step). `examples/webapp/index.html` is
  the standalone mockup it derives from.
- Avatar: `static/glados-rig.js` (original rig), `static/glados-avatar.js`
  (telemetry and lifecycle), and `static/glados-avatar.css` under the webapp package.
- Server: `src/glados/webapp/server.py`; serializers: `serializers.py`;
  config: `config.py`.
- Tests: `tests/test_webapp.py` — bus fan-out, serializers, and an in-process
  HTTP smoke test against a stub engine.
- Example config: `configs/glados_webapp_config.yaml`.

## Decision lists and speech routing

Settings supports creating, editing, reordering and deleting options and lists,
selecting the active list, enabling/disabling routing, thresholds and fallback.
An option binds to ignore, reply, clarify, general tool planning, or a specific
tool with fixed arguments. Tool controls come from the live tool schemas.
There are at most 20 options per list and 32 lists. Light actions require a
connected lighting tool; the console does not invent devices.

Settings are stored atomically in ignored `data/decision_lists.yaml`; unlike
tasks they survive restart. List/option IDs remain stable when reordered.
Edits use revision checks. An in-flight classifier retains its original list
snapshot, but a fixed action is cancelled if its list changed, was disabled or
deleted before dispatch. The tool executor checks again before starting.
Already-started physical actions cannot be undone by editing a list.

The llama.cpp adapter validates single-token labels, disables thinking, requests
one output token, and reads scores for every option. Equal positive logit bias
exposes the candidates; renormalizing over them cancels that shared bias.
Top-k/top-p/min-p/penalty sampling is disabled. Incomplete or non-finite scores
never authorize a fixed action; missing candidates never become zero-confidence
options. Scores are relative preferences, not calibrated correctness estimates.

The default routes completed turns before generating speech. Native audio goes
directly to the classifier without transcription; extended mode uses its
transcript. By default, a low-confidence classification delegates the original
text or audio to the full assistant with read-only tools (`get_time` for named timezones,
`run_safe_command`, `get_report`, `get_preferences`, and `vision_look` when the
camera mind is enabled). The original threshold and
margin still apply to fixed actions. This avoids treating classifier uncertainty
as missing user intent. Unavailable routing also uses this restricted path.
Only offered tool names can be dispatched, and fallback tool results disable
further tool calls for the reply. Settings offers this fallback as “Let GLaDOS
answer (read-only tools)”; existing saved settings can select it explicitly.
The clarification option now lets the assistant assess the original request and
ask a specific question when necessary, instead of playing a canned sentence.
Typed input cannot be silently
ignored as background speech. A selected fixed tool bypasses generated tool JSON
and its result is passed to GLaDOS with further tool calls disabled.

`docs/benchmarks/routing-assist-2026-10-05.json` records local E4B checks for
spoken/typed small talk and CPU queries with transcripts disabled. A separate
`routing-assist-fallback-2026-10-05.json` run deliberately sets the threshold to
1.0 to exercise the full assistant's read-only CPU tool call rather than the
fixed shortcut. These are generated speech-text checks, excluding TTS playback.

**Parallel drafts** are managed automatically. After the router has its slot,
a draft may acquire another free slot without queueing. Drafting is skipped on
single-slot servers, Ollama endpoints and untranslated native audio, avoiding
duplicated audio encoding. Optional transcription finishes first, allowing its
text to be reused. The stream stays in a bounded private buffer and has only
the normal reply's read-only visual tool when applicable. Only a reply decision
and fresh Emotion Core instructions release the stream to speech. If the current
interaction changes the emotion context, the draft is discarded and a new reply
uses the updated affect. Other routing decisions
cancel the draft using its own event, never the shared engine event. The producer
releases its lease when HTTP streaming ends; cancelled prompt processing may
retain capacity until the server yields or the request times out.
Optional transcription holds its own lease and finishes before routing or drafts
start, so it cannot exceed the shared capacity. Its HTTP stream closes when the
interaction is interrupted. An experiment with deferred transcription was slower
on this machine; see [Gemma evaluation](gemma4.md#optional-user-transcripts).
No periodic user-interruption
loop is implemented.

The small live test found parallel drafts improved one conversational turn by
about 0.09 s but slowed one clock-tool turn by about 0.15 s, with router latency
increasing under contention. These are observed completion windows with voice
muted, not first-audio latency. These historical checks predate automatic
draft management and emotion freshness checks. Ten text cases plus one synthetic spoken clock question selected the
expected action in ten cases; a hypothetical light command fell back to
clarification. Natural speech and broader prompt calibration remain necessary.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/memory` | Read-only saved memory and current recall. Supports `query`, `kind=all\|fact\|summary`, `offset` and `limit` (maximum 50). |
| GET | `/api/context` | Neural Buffer: ordered preview or submitted request, using `mode=user\|autonomy` and `view=live\|request`. |
| GET | `/api/decisions` | Versioned lists and activation settings. |
| POST | `/api/decisions` | `action`: `save`, `delete`, or `activate`, with the current global `revision`. |
| POST | `/api/decisions/test` | Preview a saved `list_id` using `text` or base64 mono 16 kHz WAV `audio` (max 30 s). Optional `spoken` treats text as spoken input. Never executes actions. |

Snapshots include `decisions`, `inference`, and `routing`; frequent state events
include current inference occupancy and the latest routing/preview scores.

### Vision mind

Enable `Glados.vision` to add **Cores → Vision Core**. Its inspector displays the
live webcam at up to 30 fps with CPU face tracking, plus E4B captions and
changes on the configured motion-aware schedule (2–5 seconds by default), measured frequency and sharpness timing. Each displayed frame includes its own face box, gaze crosshair and normalized
X/Y coordinates. Live video and slower captions update independently. Pause/resume controls release and reopen the camera; run-once
also works while paused. **Recent events** describes up to four timestamped
observations, while **What I see** describes the current image. The camera endpoint is independent of the speaking
model, so API conversation profiles still use E4B for vision. See [Vision](vision.md).

The main avatar and its in-flight counter show user-facing inference only.
Continuous Vision and other background minds remain visible in their lanes
and Cores → Shared processing without making the avatar say "Processing". Specific
visual questions use `vision_look(question=...)` for a fresh E4B inspection;
the Vision inspector shows the latest question, answer, and visible evidence.

Enrichment Center's **Central Core** controls include **Microphone**, **Voice**, and
**Camera: ON/OFF**. Camera toggles the Vision mind's pause/resume control and
releases the webcam when turned off. Optional voice transcripts are controlled
in **Facility Settings → Voice input** instead of the main page.


Quiet mode and emotional replies

The Central Core panel and Facility Settings provide a Quiet control. Direct requests such as “go to sleep”, “be quiet”, or “stop replying until I tell you” enter quiet mode through one-token routing. A second small choice confirms a sleep request, including negation handling. While quiet, only the microphone/input queue and a two-option wake classifier remain active. Ordinary questions, insults and background sounds do not wake her. “Wake up” or the Wake button restores the prior paused state of each core. Speech already playing is stopped, queued audio and tools are discarded, and generation markers reject late results after a sleep/wake cycle. The microphone retains its existing ON/OFF setting.

Noise, non-speech audio and unintelligible speech are routed to Ignore before the PAD update or reply generation. With optional transcripts enabled, classification uses the shared transcript when available. This is model classification, so real-room audio still benefits from tuning the options and thresholds.

For accepted user turns, Emotion updates PAD before the reply request. Current PAD adds explicit tone and starting expression instructions to the reply prompt. Insults can produce an angry glare and sharp sarcasm, while normal factual questions do not erase an existing reaction. One-second ticks decay toward neutral using elapsed time, without an LLM inference.
Decay is applied before new reactions as well, so ordinary activity does not prevent recovery.
While cores are suspended the timer stops publishing; the next tick or wake reaction catches up for elapsed time. The live Central Core panel shows the same affect instructions used for the reply.

Autonomy and conversation compaction

Facility Settings → Autonomy Core controls proactive responses at runtime, including profiles that start with autonomy OFF. Camera, Emotion and Compaction work independently. Turning autonomy OFF invalidates active and queued autonomous decisions, continuations and speech while preserving user replies. Work blocked inside model prefill or a non-cancellable tool can finish, but its stale output is rejected. Quiet mode pauses those cores as well.

The setting displays the review/handoff stage, pending notifications, last decision,
Central Core instruction/source IDs and time until the next idle check. All independent
slots are reviewed with conversation evidence. Notifications survive user activity;
stable condition keys and a delivery ledger suppress repeated announcements. Checks
coalesce automatically, back off after silence and keep internal requests private.
Requested searches run as background tasks when Autonomy is enabled; completed reports
can prompt Central Core. Fresh Vision arrival evidence can justify one greeting. See
[Autonomy Core](autonomy.md) for scheduling and the validated JSON protocol.

The Compaction Mind retains at least the last 8 messages and complete tool exchanges. Older history becomes timestamped summaries in non-overlapping bands: the last hour, 1–4 hours, 4–8 hours, 8–24 hours, 1–3 days, 3–7 days, 1–4 weeks and older. Summaries merge and move to older bands over time. Compaction runs quietly on E4B, defers to interactive work, preserves concurrent appends and leaves originals intact on failure. Voice turns with transcripts disabled preserve their available placeholder/action summary rather than a full transcript.

The local E4B profile persists timestamped conversation records in `~/.glados/conversation.json` with owner-only permissions; personality/system prompts come from the current config. The Mind inspector shows context estimates and band counts.
