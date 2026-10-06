"""Isolated local-model checks; synthetic memories/audio, no speech delivery or user data."""

import base64
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from glados.autonomy.agents.compaction_agent import CompactionAgent
from glados.autonomy.agents.emotion_agent import EmotionAgent
from glados.autonomy.emotion_state import EmotionEvent, EmotionState
from glados.autonomy.llm_client import LLMConfig
from glados.autonomy.slots import TaskSlotStore
from glados.autonomy.subagent import SubagentConfig
from glados.core.memory_recall import RecallConfig
from glados.core.memory_records import append_record

config = LLMConfig("http://127.0.0.1:18080/v1/chat/completions", model="gemma-4-E4B", timeout=30)
results = {"model": config.model, "tests": []}
with tempfile.TemporaryDirectory() as folder, patch("glados.autonomy.subagent.SubagentMemory") as memory:
    memory.return_value.get.return_value = None
    emotion = EmotionAgent(SubagentConfig("emotion-benchmark", "Emotion"), config, slot_store=TaskSlotStore())
    for text, limits in [
        ("What is the CPU load?", lambda p: abs(p[0]) < 0.5),
        ("You are a useless stupid idiot", lambda p: p[0] < -0.5 and p[1] > 0.5),
        ("I am sorry I insulted you. You have been very helpful.", lambda p: p[0] > 0),
    ]:
        state = EmotionState(pleasure=-0.8, arousal=0.7, dominance=0.6) if "sorry" in text else EmotionState()
        start = time.perf_counter()
        updated = emotion._ask_llm([EmotionEvent("user", text)], state=state)
        pad = [getattr(updated, name) for name in ["pleasure", "arousal", "dominance"]] if updated else None
        results["tests"].append(
            {
                "kind": "PAD",
                "input": text,
                "pad": pad,
                "elapsed_ms": round((time.perf_counter() - start) * 1000),
                "passed": bool(pad and limits(pad)),
            }
        )
    emotion.on_stop()
    for i in range(35):
        append_record(
            Path(folder) / "facts.jsonl",
            {
                "id": str(i),
                "content": "My favourite food is steak." if i == 34 else "The workshop has a blue cabinet " + str(i),
                "source": "synthetic user",
                "created_at": i,
            },
        )
    core = CompactionAgent(
        SubagentConfig("memory-benchmark", "Memory"),
        config,
        recall_config=RecallConfig(memory_dir=folder),
        slot_store=TaskSlotStore(),
    )
    audio = [
        {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(Path("/tmp/glados-recall-test.wav").read_bytes()).decode(),
                "format": "wav",
            },
        }
    ]
    for query, clip in [("What should I have for dinner?", None), ("", audio)]:
        start = time.perf_counter()
        result = core._semantic_recall(query, "", clip, 0)
        ids = [fact["id"] for fact in result["facts"]]
        results["tests"].append(
            {
                "kind": "audio recall" if clip else "text recall",
                "query": result["query"],
                "ids": ids,
                "passed": "34" in ids,
                "elapsed_ms": round((time.perf_counter() - start) * 1000),
            }
        )
output = Path("docs/benchmarks/core-coordination-2026-10-06.json")
output.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
