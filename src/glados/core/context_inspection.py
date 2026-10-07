"""Readable snapshots of inference context; binary attachments stay transient."""

from __future__ import annotations

import time
from typing import Any

TITLES = {
    "system": "Base personality", "clock": "Current local time and date · host clock",
    "speech": "Speech and animation format", "console": "Tool and task rules",
    "autonomy": "Autonomy Core instructions", "operator": "Session preferences · language and style",
    "autonomy_decision": "Autonomy Core · null or Central Core handoff",
    "autonomy_chat": "Autonomy Core · chat summaries and recent turns",
    "autonomy_slots": "Autonomy Core · all core and task slots",
    "emotion": "Emotion Core · PAD and response tone", "preferences": "User preferences",
    "slots": "Task assignments and job results", "memory": "Retrieved memory",
    "knowledge": "Knowledge", "constitution": "Personality modifiers",
    "mcp": "Connected service data", "vision": "Vision Core · current observation",
    "health": "Health Core · system readings and rolling commentary",
    "history": "Chat history", "summary": "Memory Core · compacted history",
    "input": "Current request input", "request": "Request and routing instructions",
}

PURPOSES = {
    "system": "Reusable personality instructions.",
    "speech": "Reusable output format: silent animation markers and speakable text.",
    "console": "Reusable rules for truthful tool use, measurements and saved tasks.",
    "operator": "Your editable session preferences, such as language, reply length and style.",
    "autonomy": "Independent reviewer policy: decide whether to request a Central Core notification.",
    "autonomy_decision": "Validated decision: stay silent or request a response from Central Core with cited slots.",
    "autonomy_chat": "Existing Memory Core summaries and the last eight text turns; quoted evidence for timing and deduplication.",
    "autonomy_slots": "All independent core and task outputs, including timestamps, condition identities and bounded reports.",
    "emotion": "Live PAD values and tone guidance for this turn; separate from session preferences.",
    "slots": "Saved assignments, useful job results and relevant facts recalled through the Memory Core slot. Compaction statistics and dedicated emotion/vision state are not repeated.",
    "clock": "Fresh host reading assembled for each response: local time, date, weekday and timezone. No tool call needed.",
    "vision": "Latest camera observation, with its freshness information.",
    "health": "Timestamped host/runtime readings, transition alerts and cached commentary; status questions need no new metric tool call.",
    "mcp": "Data from explicitly configured service resources; this is not the tool catalogue.",
    "history": "Conversation messages in their original order, including tool calls and results.",
    "summary": "Compacted conversation facts, replacing older history.",
    "input": "The latest user input or current tool continuation actually sent to the model.",
    "request": "Instructions and routing or tool-result data specific to this request.",
    "preferences": "Saved user preferences.", "memory": "Memory retrieved for this request.",
    "knowledge": "Saved knowledge.", "constitution": "Personality modifiers.",
}


def readable_payload(value: Any) -> Any:
    """Copy textual content, describing media without copying its encoded bytes."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "input_audio" and isinstance(item, dict):
                result[key] = {"format": item.get("format"), "data": "[audio payload omitted]"}
            elif key == "image_url":
                if isinstance(item, dict):
                    result[key] = {**readable_payload(item), "url": "[media payload omitted]"} if str(item.get("url", "")).startswith("data:") else readable_payload(item)
                else:
                    result[key] = "[media payload omitted]" if str(item).startswith("data:") else item
            elif key == "images" and isinstance(item, list):
                result[key] = ["[image payload omitted]" for _ in item]
            else:
                result[key] = readable_payload(item)
        return result
    if isinstance(value, (list, tuple)):
        return [readable_payload(item) for item in value]
    return value


def describe_context(
    messages: list[dict[str, Any]], sources: list[str], tools: list[dict[str, Any]],
    *, model: str, mode: str, kind: str,
) -> dict[str, Any]:
    """Keep exact message order and group adjacent messages from the same source."""
    sections: list[dict[str, Any]] = []
    for index, (message, source) in enumerate(zip(messages, sources, strict=True)):
        if source == "history" and str(message.get("content", "")).startswith("[summary]"):
            source = "summary"
        if not sections or sections[-1]["source"] != source:
            sections.append({"source": source, "title": TITLES.get(source, source.replace("_", " ").title()),
                             "description": PURPOSES.get(source, ""), "messages": []})
        sections[-1]["messages"].append({**readable_payload(message), "index": index + 1})
    characters = sum(len(m["content"]) for section in sections for m in section["messages"]
                     if isinstance(m.get("content"), str))
    return {"available": True, "captured_at": time.time(), "kind": kind, "mode": mode, "model": model,
            "message_count": len(messages), "text_characters": characters,
            "sections": sections, "tools": readable_payload(tools)}
