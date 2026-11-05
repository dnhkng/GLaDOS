"""Console slash commands (/help, /status, ...), mixed into :class:`~glados.core.engine.Glados`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING, Any, Literal

from ..autonomy.summarization import estimate_tokens

if TYPE_CHECKING:
    from ..autonomy import AutonomyConfig, ConstitutionalState, MindScheduler, TaskSlotStore
    from ..autonomy.agents import EmotionAgent
    from ..mcp import MCPManager
    from ..observability import MindRegistry
    from ..vision import VisionState
    from .conversation_store import ConversationStore
    from .inference import InferenceScheduler
    from .knowledge_store import KnowledgeStore
    from .store import Store


@dataclass(frozen=True)
class CommandSpec:
    name: str
    description: str
    handler: Callable[[list[str]], str]
    usage: str | None = None
    aliases: tuple[str, ...] = ()


class GladosCommands:
    """Command registry and handlers; relies on the engine state set up by ``Glados.__init__``."""

    # The engine state these commands read and control, provided by the host class (Glados).
    _command_registry: dict[str, CommandSpec]
    _command_order: list[str]
    _conversation_store: ConversationStore
    _emotion_agent: EmotionAgent | None
    _quiet_generation: int
    asr_muted_event: threading.Event
    autonomy_config: AutonomyConfig
    autonomy_slots: TaskSlotStore | None
    constitutional_state: ConstitutionalState
    inference_scheduler: InferenceScheduler
    input_mode: Literal["audio", "text", "both"]
    knowledge_store: KnowledgeStore
    mcp_manager: MCPManager | None
    mind_registry: MindRegistry
    preferences_store: Store[Any]
    shutdown_event: threading.Event
    subagent_manager: MindScheduler | None
    tts_muted_event: threading.Event
    vision_state: VisionState | None

    if TYPE_CHECKING:

        def set_asr_muted(self, muted: bool) -> None: ...
        def set_tts_muted(self, muted: bool) -> None: ...
        def set_autonomy_enabled(self, enabled: bool) -> None: ...

    def command_specs(self) -> list[CommandSpec]:
        return [self._command_registry[name] for name in self._command_order]

    def handle_command(self, command: str) -> str:
        text = command.strip()
        if not text:
            return "No command entered."
        text = text.removeprefix("/")
        parts = text.split()
        if not parts:
            return "No command entered."
        cmd = parts[0].lower()
        args = parts[1:]
        spec = self._command_registry.get(cmd)
        if not spec:
            return f"Unknown command: /{cmd}. Try /help."
        reply: str = spec.handler(args)
        return reply

    def _build_command_registry(self) -> tuple[dict[str, CommandSpec], list[str]]:
        registry: dict[str, CommandSpec] = {}
        order: list[str] = []

        def register(spec: CommandSpec) -> None:
            registry[spec.name] = spec
            order.append(spec.name)
            for alias in spec.aliases:
                registry[alias] = spec

        register(
            CommandSpec(
                name="help",
                description="Show available commands",
                usage="/help",
                handler=self._cmd_help,
                aliases=("?",),
            )
        )
        register(
            CommandSpec(
                name="status",
                description="Show engine status",
                usage="/status",
                handler=self._cmd_status,
            )
        )
        register(
            CommandSpec(
                name="tts",
                description="Control TTS output",
                usage="/tts on|off",
                handler=self._cmd_tts,
            )
        )
        register(
            CommandSpec(
                name="quit",
                description="Quit GLaDOS",
                usage="/quit",
                handler=self._cmd_quit,
                aliases=("exit",),
            )
        )
        register(
            CommandSpec(
                name="asr",
                description="Control ASR input",
                usage="/asr on|off",
                handler=self._cmd_asr,
            )
        )
        register(
            CommandSpec(
                name="observe",
                description="Open observability screen (TUI)",
                usage="/observe",
                handler=self._cmd_observe,
                aliases=("observability",),
            )
        )
        register(
            CommandSpec(
                name="mcp",
                description="Show MCP server status",
                usage="/mcp status",
                handler=self._cmd_mcp,
            )
        )
        register(
            CommandSpec(
                name="autonomy",
                description="Manage autonomy settings",
                usage="/autonomy on|off",
                handler=self._cmd_autonomy,
            )
        )
        register(
            CommandSpec(
                name="slots",
                description="Show autonomy slots",
                usage="/slots",
                handler=self._cmd_slots,
            )
        )
        register(
            CommandSpec(
                name="minds",
                description="Show active minds",
                usage="/minds",
                handler=self._cmd_minds,
            )
        )
        register(
            CommandSpec(
                name="agents",
                description="Show registered subagents",
                usage="/agents",
                handler=self._cmd_agents,
            )
        )
        register(
            CommandSpec(
                name="emotion",
                description="Show current emotional state",
                usage="/emotion",
                handler=self._cmd_emotion,
            )
        )
        register(
            CommandSpec(
                name="preferences",
                description="Show user preferences",
                usage="/preferences",
                handler=self._cmd_preferences,
            )
        )
        register(
            CommandSpec(
                name="context",
                description="Show context/token usage",
                usage="/context",
                handler=self._cmd_context,
            )
        )
        register(
            CommandSpec(
                name="constitution",
                description="Show constitutional state and modifiers",
                usage="/constitution",
                handler=self._cmd_constitution,
            )
        )
        register(
            CommandSpec(
                name="vision",
                description="Show latest vision snapshot",
                usage="/vision",
                handler=self._cmd_vision,
            )
        )
        register(
            CommandSpec(
                name="config",
                description="Show config summary",
                usage="/config",
                handler=self._cmd_config,
            )
        )
        register(
            CommandSpec(
                name="knowledge",
                description="Manage local knowledge notes",
                usage="/knowledge add|list|set|delete|clear",
                handler=self._cmd_knowledge,
            )
        )
        register(
            CommandSpec(
                name="memory",
                description="Show long-term memory stats",
                usage="/memory",
                handler=self._cmd_memory,
            )
        )
        return registry, order

    def _cmd_help(self, _args: list[str]) -> str:
        lines = ["Commands:"]
        for name in self._command_order:
            spec = self._command_registry[name]
            usage = spec.usage or f"/{spec.name}"
            lines.append(f"- {usage}: {spec.description}")
        return "\n".join(lines)

    def _cmd_status(self, _args: list[str]) -> str:
        autonomy_enabled = self.autonomy_config.enabled
        vision_enabled = self.vision_state is not None
        jobs_enabled = bool(self.autonomy_config.jobs.enabled) if self.autonomy_config else False
        return (
            f"input_mode={self.input_mode}, "
            f"asr_muted={self.asr_muted_event.is_set()}, "
            f"tts_muted={self.tts_muted_event.is_set()}, "
            f"autonomy_enabled={autonomy_enabled}, "
            f"vision_enabled={vision_enabled}, "
            f"jobs_enabled={jobs_enabled}"
        )

    def _cmd_quit(self, _args: list[str]) -> str:
        self.inference_scheduler.end_interaction(self._quiet_generation, "shutdown")
        self.shutdown_event.set()
        return "Shutting down."

    def _cmd_asr(self, args: list[str]) -> str:
        if not args:
            return f"ASR is {'muted' if self.asr_muted_event.is_set() else 'active'}."
        arg = args[0].lower()
        if arg in {"on", "unmute", "active"}:
            self.set_asr_muted(False)
            return "ASR unmuted."
        if arg in {"off", "mute"}:
            self.set_asr_muted(True)
            return "ASR muted."
        return "Usage: /asr on|off"

    def _cmd_tts(self, args: list[str]) -> str:
        if not args:
            return f"TTS is {'muted' if self.tts_muted_event.is_set() else 'active'}."
        arg = args[0].lower()
        if arg in {"on", "unmute", "active"}:
            self.set_tts_muted(False)
            return "TTS unmuted."
        if arg in {"off", "mute"}:
            self.set_tts_muted(True)
            return "TTS muted."
        return "Usage: /tts on|off"

    def _cmd_observe(self, _args: list[str]) -> str:
        return "Observability is available in the TUI via /observe."

    def _cmd_slots(self, _args: list[str]) -> str:
        if not self.autonomy_slots:
            return "Autonomy slots are unavailable."
        slots = self.autonomy_slots.list_slots()
        if not slots:
            return "No active slots."
        lines = ["Slots:"]
        for slot in slots[:20]:
            summary = slot.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            lines.append(f"- [{slot.slot_id}] {slot.title}: {slot.status}{summary_text}")
        if len(slots) > 20:
            lines.append(f"... {len(slots) - 20} more")
        return "\n".join(lines)

    def _cmd_minds(self, _args: list[str]) -> str:
        minds = self.mind_registry.snapshot()
        if not minds:
            return "No minds registered."
        lines = ["Minds:"]
        for mind in minds[:20]:
            summary = mind.summary.strip()
            summary_text = f" - {summary}" if summary else ""
            lines.append(f"- {mind.title}: {mind.status}{summary_text}")
        if len(minds) > 20:
            lines.append(f"... {len(minds) - 20} more")
        return "\n".join(lines)

    def _cmd_agents(self, _args: list[str]) -> str:
        if not self.subagent_manager:
            return "Subagent manager is not enabled."
        agents = self.subagent_manager.list_agents()
        if not agents:
            return "No subagents registered."
        lines = ["Subagents:"]
        for agent in agents[:20]:
            status = "running" if agent.running else "stopped"
            tick_info = f"ticks={agent.tick_count}" if agent.tick_count > 0 else "not started"
            lines.append(f"- {agent.title} ({agent.agent_id}): {status}, {tick_info}")
        if len(agents) > 20:
            lines.append(f"... {len(agents) - 20} more")
        return "\n".join(lines)

    def _cmd_emotion(self, _args: list[str]) -> str:
        if not self._emotion_agent:
            return "Emotion agent is not running."
        state = self._emotion_agent.state
        lines = [
            "Emotional State:",
            f"  Pleasure:  {state.pleasure:+.2f}",
            f"  Arousal:   {state.arousal:+.2f}",
            f"  Dominance: {state.dominance:+.2f}",
            "Mood Baseline:",
            f"  Pleasure:  {state.mood_pleasure:+.2f}",
            f"  Arousal:   {state.mood_arousal:+.2f}",
            f"  Dominance: {state.mood_dominance:+.2f}",
            "",
            state.to_prompt(),
        ]
        return "\n".join(lines)

    def _cmd_preferences(self, _args: list[str]) -> str:
        prefs = self.preferences_store.all()
        if not prefs:
            return "No preferences set."
        lines = ["User Preferences:"]
        for key, value in prefs.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def _cmd_context(self, _args: list[str]) -> str:
        messages = self._conversation_store.snapshot()
        token_count = estimate_tokens(messages)
        msg_count = len(messages)
        system_count = sum(1 for m in messages if m.get("role") == "system")
        user_count = sum(1 for m in messages if m.get("role") == "user")
        assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
        summary_count = sum(
            1 for m in messages if isinstance(m.get("content"), str) and m["content"].startswith("[summary]")
        )
        lines = [
            "Context Usage:",
            f"  Estimated tokens: {token_count}",
            f"  Total messages: {msg_count}",
            f"    System: {system_count}",
            f"    User: {user_count}",
            f"    Assistant: {assistant_count}",
            f"    Summaries: {summary_count}",
        ]
        return "\n".join(lines)

    def _cmd_constitution(self, _args: list[str]) -> str:
        state = self.constitutional_state
        lines = ["Constitutional State:"]
        lines.append("")
        lines.append("Immutable Rules:")
        for rule in state.constitution.immutable_rules:
            lines.append(f"  - {rule}")
        lines.append("")
        lines.append("Modifiable Bounds:")
        for name, (min_val, max_val) in state.constitution.modifiable_bounds.items():
            lines.append(f"  {name}: {min_val} to {max_val}")
        lines.append("")
        if state.active_modifiers:
            lines.append("Active Modifiers:")
            for name, modifier in state.active_modifiers.items():
                lines.append(f"  {name}: {modifier.value} ({modifier.reason})")
        else:
            lines.append("Active Modifiers: none")
        lines.append("")
        lines.append(f"Modifier History: {len(state.modifier_history)} changes")
        return "\n".join(lines)

    def _cmd_vision(self, _args: list[str]) -> str:
        if not self.vision_state:
            return "Vision is disabled."
        snapshot = self.vision_state.snapshot()
        return snapshot or "Vision has no snapshot yet."

    def _cmd_mcp(self, args: list[str]) -> str:
        if not self.mcp_manager:
            return "MCP is disabled."
        if args and args[0].lower() not in {"status", "list"}:
            return "Usage: /mcp status"
        lines = ["MCP servers:"]
        for entry in self.mcp_manager.status_snapshot():
            status = "connected" if entry["connected"] else "offline"
            tools = entry.get("tools", 0)
            resources = entry.get("resources", 0)
            lines.append(f"- {entry['name']}: {status}, tools={tools}, resources={resources}")
        return "\n".join(lines)

    def _cmd_autonomy(self, args: list[str]) -> str:
        if not args:
            return (
                f"Autonomy enabled={self.autonomy_config.enabled}, "
                f"parallel_calls={self.autonomy_config.autonomy_parallel_calls}, "
                "coalescing=automatic"
            )
        head = args[0].lower()
        if head in {"on", "off", "true", "false", "enable", "enabled", "disable", "disabled"}:
            enabled = head in {"on", "true", "enable", "enabled"}
            self.set_autonomy_enabled(enabled)
            return f"Autonomy {'enabled' if enabled else 'disabled'}."
        return "Usage: /autonomy on|off (idle checks coalesce automatically)"

    def _cmd_config(self, _args: list[str]) -> str:
        jobs_enabled = bool(self.autonomy_config.jobs.enabled) if self.autonomy_config else False
        return (
            f"input_mode={self.input_mode}, "
            f"autonomy.enabled={self.autonomy_config.enabled}, "
            f"autonomy.jobs.enabled={jobs_enabled}, "
            f"vision.enabled={self.vision_state is not None}"
        )

    def _cmd_knowledge(self, args: list[str]) -> str:
        if not args or args[0] == "list":
            entries = self.knowledge_store.list_entries()
            if not entries:
                return "Knowledge: no entries."
            lines = ["Knowledge:"]
            for entry in entries[:20]:
                text = entry.text.strip()
                preview = (text[:120] + "...") if len(text) > 120 else text
                lines.append(f"- {entry.entry_id}: {preview}")
            if len(entries) > 20:
                lines.append(f"... {len(entries) - 20} more")
            return "\n".join(lines)

        action = args[0].lower()
        if action == "add":
            text = " ".join(args[1:]).strip()
            if not text:
                return "Usage: /knowledge add <text>"
            entry = self.knowledge_store.add(text)
            return f"Added knowledge #{entry.entry_id}."

        if action in {"set", "update"}:
            if len(args) < 3:
                return "Usage: /knowledge set <id> <text>"
            try:
                entry_id = int(args[1])
            except ValueError:
                return "Knowledge id must be a number."
            text = " ".join(args[2:]).strip()
            if not text:
                return "Usage: /knowledge set <id> <text>"
            updated = self.knowledge_store.update(entry_id, text)
            if not updated:
                return f"Knowledge #{entry_id} not found."
            return f"Updated knowledge #{entry_id}."

        if action in {"delete", "remove"}:
            if len(args) < 2:
                return "Usage: /knowledge delete <id>"
            try:
                entry_id = int(args[1])
            except ValueError:
                return "Knowledge id must be a number."
            removed = self.knowledge_store.delete(entry_id)
            if not removed:
                return f"Knowledge #{entry_id} not found."
            return f"Deleted knowledge #{entry_id}."

        if action == "clear":
            cleared = self.knowledge_store.clear()
            return f"Cleared {cleared} knowledge entr{'y' if cleared == 1 else 'ies'}."

        return "Usage: /knowledge add|list|set|delete|clear"

    def _cmd_memory(self, _args: list[str]) -> str:
        import json
        from pathlib import Path

        memory_dir = Path.home() / ".glados" / "memory"
        facts_file = memory_dir / "facts.jsonl"
        summaries_file = memory_dir / "summaries.jsonl"

        # Count facts
        fact_count = 0
        source_counts: dict[str, int] = {}
        total_importance = 0.0
        if facts_file.exists():
            try:
                with facts_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            fact_count += 1
                            try:
                                fact = json.loads(line)
                                source = fact.get("source", "unknown")
                                source_counts[source] = source_counts.get(source, 0) + 1
                                total_importance += fact.get("importance", 0.5)
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass

        # Count summaries
        summary_count = 0
        period_counts: dict[str, int] = {}
        if summaries_file.exists():
            try:
                with summaries_file.open("r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            summary_count += 1
                            try:
                                summary = json.loads(line)
                                period = summary.get("period", "unknown")
                                period_counts[period] = period_counts.get(period, 0) + 1
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass

        if fact_count == 0 and summary_count == 0:
            return "Long-term Memory: empty (no facts or summaries stored)"

        avg_importance = total_importance / fact_count if fact_count > 0 else 0.0

        lines = [
            "Long-term Memory Stats:",
            f"  Total facts: {fact_count}",
            f"  Total summaries: {summary_count}",
        ]

        if source_counts:
            lines.append("  Facts by source:")
            for source, count in sorted(source_counts.items()):
                lines.append(f"    {source}: {count}")

        if period_counts:
            lines.append("  Summaries by period:")
            for period, count in sorted(period_counts.items()):
                lines.append(f"    {period}: {count}")

        lines.append(f"  Average importance: {avg_importance:.2f}")
        lines.append(f"  Storage: {memory_dir}")

        return "\n".join(lines)
