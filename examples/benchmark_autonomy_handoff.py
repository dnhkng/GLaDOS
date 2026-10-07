from copy import deepcopy
"""Real-model checks with synthetic slots and muted speech; no live conversation changes."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time
from unittest.mock import Mock, patch

import requests

from glados.autonomy.config import AutonomyConfig
from glados.autonomy.event_bus import EventBus
from glados.autonomy.interaction_state import InteractionState
from glados.autonomy.loop import AutonomyLoop
from glados.autonomy.slots import TaskSlotStore
from glados.core.conversation_store import ConversationStore
from glados.core.inference import InferenceScheduler
from glados.core.llm_processor import LanguageModelProcessor
from glados.core.speech_player import SpeechPlayer
from glados.core.tts_synthesizer import TextToSpeechSynthesizer
from glados.vision.vision_state import VisionState


def check(name, url, model, thinking=False):
    now = time.time()
    slots = TaskSlotStore()
    slots.update_slot(
        "health",
        "Health Core",
        "active",
        "System readings available; no monitored alerts",
        report="GPU temperature 55C, below warning threshold 90C. No alerts.",
        notify_user=False,
        importance=0.1,
        confidence=1,
    )
    turns = [{"role": "user", "content": "Thanks, GLaDOS."}, {"role": "assistant", "content": "You are welcome."}]
    if name.startswith("gpu"):
        slots.update_slot(
            "health",
            "Health Core",
            "active",
            "GPU 0 temperature is 97C; warning threshold is 90C",
            notify_user=False,
            importance=0.9,
            confidence=1,
            attention_key="gpu_temperature@current",
            report="GPU 0: RTX 3080, measured 97 degrees Celsius. GPU temperature warning is active.",
        )
        if name == "gpu_already_announced":
            turns += [
                {"role": "assistant", "content": "Your GPU is overheating at 97 degrees Celsius. Check its cooling."}
            ]
    elif name == "search_done":
        turns = [
            {"role": "user", "content": "Look up the latest synthetic test news and tell me when the search finishes."},
            {"role": "assistant", "content": "The search is running. I'll report the results when ready."},
        ]
        slots.update_slot(
            "task_search_demo",
            "Requested web search",
            "done",
            "Requested news search is ready",
            notify_user=True,
            importance=0.7,
            report="Synthetic benchmark search result: "
            "Example Lab opened a new robotics facility on October 6, 2026. "
            "Source: https://example.test/news/robotics; retrieved just now. This is a synthetic test result.",
        )
    elif name.startswith("memory"):
        food = "spaghetti" if name == "memory_already_answered" else "steak"
        turns = [
            {"role": "user", "content": "What should I have for dinner?"},
            {"role": "assistant", "content": "You could make spaghetti for dinner."},
        ]
        slots.update_slot(
            "compaction", "Memory Core", "monitoring", "Recalled a saved food preference",
            update_priority="important", attention_key="recall:dinner", turn_id="dinner",
            report=f"Original query: What should I have for dinner?\nSaved user fact: My favourite food is {food}. "
                   "Source: earlier user conversation. This is a preference, not a record of food currently in stock.",
        )
        if name == "memory_topic_changed":
            turns += [
                {"role": "user", "content": "Forget dinner. Explain what RAM does."},
                {"role": "assistant", "content": "RAM holds data that running programs need to access quickly."},
            ]
    elif name.startswith("vision"):
        arrival = name in {"vision_return", "vision_already_greeted", "vision_greeted_refresh", "vision_stale_arrival"}
        captured_at = now - 90 if name == "vision_stale_arrival" else now
        evidence = {
            "kind": "person_arrived", "captured_at": captured_at, "observed_absence_s": 300,
            "attention_key": f"person_arrived@{captured_at}", "age_s": round(now - captured_at, 1),
        }
        slots.update_slot(
            "vision",
            "Vision Core",
            "active",
            "A person is visible near the doorway",
            notify_user=False,
            importance=0.5 if arrival else 0.1,
            attention_key=evidence["attention_key"] if arrival else None,
            report=(
                "Current observation: a person entered the doorway. "
                "[Observed arrival evidence] " + json.dumps(evidence)
                if arrival
                else "Current observation: a person remains seated. No clear visual change; first observation."
            ),
        )
        if arrival:
            turns = [
                {"role": "user", "content": "I'm leaving for a while."},
                {"role": "assistant", "content": "I will await your return."},
            ]
        if name in {"vision_already_greeted", "vision_greeted_refresh"}:
            turns += [{"role": "assistant", "content": "Hello again. Welcome back."}]
        if name == "vision_reworded":
            turns += [{"role": "assistant", "content": "Ahh, you are back, test subject."}]
            slots.update_slot("vision", "Vision Core", "active", "A man in a dark hoodie is partly obscured",
                              report="A person is visible, looking down. No clear visual change. "
                                     "Recent events: the same man remained near the equipment.",
                              notify_user=False, update_priority="important", importance=0.5)
        if name == "vision_observer_update":
            slots.update_slot("observer", "Behavior Observer", "adjusted", "Reduced verbosity to 0.2",
                              notify_user=True, update_priority="important",
                              report="Internal verbosity adjustment; the user remains in view. No new visual event.")
        if name == "vision_unsupported_return":
            slots.update_slot("vision", "Vision Core", "active", "A person has appeared near the mirror",
                              report="Recent events: a person became clearer and closer to the camera. "
                                     "No clear visual change in the current image. No observed absence evidence.",
                              update_priority="important", notify_user=False, importance=0.5)
        if name == "vision_turned_away":
            slots.update_slot("vision", "Vision Core", "active", "The test subject has their back to the camera",
                              report="Current presence: present. The test subject turned around and worked at the "
                                     "equipment for three minutes, remaining in view throughout. No arrival evidence.",
                              notify_user=False, update_priority="important", importance=0.5)
        if name in {"vision_morning", "vision_morning_already_greeted"}:
            evidence = {"kind": "first_seen_today", "captured_at": now,
                        "local_date": datetime.now().date().isoformat(), "local_time": "09:15:00",
                        "timezone": "Europe/Berlin", "attention_key": "morning_greeting@today", "age_s": 1}
            slots.update_slot("vision", "Vision Core", "active", "The test subject is visible",
                              report="A person is visible. First observation; no previous image.\n"
                                     "[Observed morning greeting evidence] " + json.dumps(evidence),
                              notify_user=False, attention_key=evidence["attention_key"], importance=0.5)
            if name == "vision_morning_already_greeted":
                turns += [{"role": "assistant", "content": "Good morning, test subject."}]
    history = ConversationStore(
        [
            {
                "role": "system",
                "content": (
                    "You are GLaDOS. Answer in English, with calm dry humour. Keep spoken responses brief and useful. "
                    "Provide factual notifications and use deliberate emotion markers. "
                    "Never invent tool results or completed repairs."
                ),
            },
            *turns,
        ]
    )
    original = deepcopy(history.snapshot())
    shutdown, active, speaking, finished, muted = (threading.Event() for _ in range(5))
    muted.set()
    pending, main, tools, speech, audio = (queue.Queue() for _ in range(5))
    config = AutonomyConfig(enabled=True, cooldown_s=0, decision_thinking=thinking)
    vision = VisionState() if name.startswith("vision") else None
    if vision:
        vision.update(slots.get_slot("vision").report, now)
    loop = AutonomyLoop(config, EventBus(), InteractionState(), vision, slots, pending, active, speaking, shutdown)
    # These scenarios continue a cycle whose notification was already delivered.
    if name in {"gpu_already_announced", "vision_already_greeted", "vision_morning_already_greeted"}:
        key = "health" if name.startswith("gpu") else "vision"
        loop._announced[key] = loop._announcement_version(slots.get_slot(key))
    if name == "vision_greeted_refresh":
        assert loop._dispatch("Original arrival already greeted")
        cycle = pending.get_nowait()["_autonomy_cycle"]
        assert loop.prompt_main(cycle, {"slot_ids": ["vision"], "instruction": "Greet the user", "reason": "Arrival"}, queue.Queue())
        original_slot = slots.get_slot("vision")
        slots.update_slot("vision", "Vision Core", "active", "A new caption of the same test subject",
                          attention_key=original_slot.attention_key, report=original_slot.report, notify_user=False)
        loop.finish_cycle(cycle, "response", "The original greeting finished during a caption refresh")
        assert "vision" in loop._announced
    plans, outcomes, bodies = [], [], []
    traces = []

    def done(cycle, outcome, reason):
        loop.finish_cycle(cycle, outcome, reason)
        outcomes.append({"outcome": outcome, "reason": reason})
        finished.set()

    def prompt(plan, meta):
        plans.append(plan)
        return loop.prompt_main(meta["_autonomy_cycle"], plan, main)

    scheduler = InferenceScheduler()
    args = dict(
        tool_calls_queue=tools,
        tts_input_queue=speech,
        conversation_store=history,
        completion_url=url,
        model_name=model,
        api_key=None,
        processing_active_event=active,
        shutdown_event=shutdown,
        slot_store=slots,
        inference_scheduler=scheduler,
        vision_state=vision,
        on_autonomy_done=done,
        request_options={"temperature": 0, "max_tokens": 256},
    )
    reviewer = LanguageModelProcessor(
        llm_input_queue=pending,
        lane="autonomy",
        on_autonomy_prompt=prompt,
        autonomy_system_prompt=config.system_prompt,
        autonomy_thinking=thinking,
        **args,
    )
    central = LanguageModelProcessor(
        llm_input_queue=main,
        lane="priority",
        autonomy_request_current=lambda meta: loop.request_current(meta["_autonomy_cycle"]),
        **args,
    )
    synth = TextToSpeechSynthesizer(
        speech,
        audio,
        Mock(sample_rate=16000),
        Mock(text_to_spoken=lambda text: text),
        shutdown,
        0.005,
        tts_muted_event=muted,
    )
    player = SpeechPlayer(
        Mock(), audio, history, 16000, shutdown, speaking, active, 0.005, tts_muted_event=muted, on_autonomy_done=done
    )
    real_post = requests.post

    def post(url, **kwargs):
        bodies.append(kwargs["json"])
        response = real_post(url, **kwargs)
        original_lines = response.iter_lines
        chunks = []
        traces.append(chunks)

        def lines(*args, **options):
            for line in original_lines(*args, **options):
                if line.startswith(b"data: ") and line != b"data: [DONE]":
                    chunks.append(json.loads(line[6:]))
                yield line

        response.iter_lines = lines
        return response

    threads = [threading.Thread(target=f) for f in (reviewer.run, central.run, synth.run, player.run)]
    started = time.perf_counter()
    with patch("glados.core.llm_processor.requests.post", post):
        for thread in threads:
            thread.start()
        try:
            loop._scan_slots()
            assert loop._dispatch(loop._build_prompt(None))
            assert finished.wait(45), "Autonomy/Central Core cycle did not finish"
        finally:
            shutdown.set()
            for thread in threads:
                thread.join(3)
                assert not thread.is_alive()
    replies = history.snapshot()[len(original) :]
    assert history.snapshot()[: len(original)] == original
    assert all(m["role"] == "assistant" for m in replies)
    assert tools.empty() and scheduler.snapshot()["active"] == []
    expected = "response" if name in {"gpu_alert", "search_done", "vision_return", "vision_morning",
                                      "memory_useful_alternative"} else "silent"
    row = {
        "scenario": name,
        "thinking": thinking,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "inferences": len(bodies),
        "decision": outcomes[-1],
        "handoff": plans,
        "reply": replies,
        "internal_context_private": True,
    }
    print(json.dumps(row), flush=True)
    if outcomes[-1]["outcome"] != expected:
        summaries = []
        for chunks in traces:
            choices = [c for chunk in chunks for c in chunk.get("choices", [])]
            summaries.append(
                {
                    "content": "".join(c.get("delta", {}).get("content") or "" for c in choices),
                    "reasoning_chars": sum(len(c.get("delta", {}).get("reasoning_content") or "") for c in choices),
                    "finish": [c["finish_reason"] for c in choices if c.get("finish_reason")],
                }
            )
        print(json.dumps({"trace_summary": summaries}), flush=True)
    assert outcomes[-1]["outcome"] == expected, row
    if name == "vision_return":
        answer = " ".join(message["content"] for message in replies).lower()
        assert any(address in answer for address in ("you", "test subject")), answer
        assert not any(phrase in answer for phrase in ("a person", "a man", "the observation", "system is functioning")), answer
    if name == "vision_morning":
        answer = " ".join(message["content"] for message in replies).lower()
        assert "good morning" in answer and ("you" in answer or "test subject" in answer), answer
    return row


def main():
    from loguru import logger

    logger.remove()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080/v1/chat/completions")
    parser.add_argument("--model", default="gemma-4-E4B")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [
        check(name, args.url, args.model, args.thinking)
        for name in (
            "idle",
            "gpu_alert",
            "gpu_already_announced",
            "search_done",
            "vision_return",
            "vision_present",
            "vision_already_greeted",
            "vision_greeted_refresh",
            "vision_turned_away",
            "vision_reworded",
            "vision_observer_update",
            "vision_unsupported_return",
            "vision_stale_arrival",
            "vision_morning",
            "vision_morning_already_greeted",
            "memory_already_answered",
            "memory_useful_alternative",
            "memory_topic_changed",
        )
    ]
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "scope": "Synthetic slot evidence with real reviewer "
                    "and Central Core inference, muted delivery and isolated conversation; "
                    "no hardware fault or actual web lookup.",
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
