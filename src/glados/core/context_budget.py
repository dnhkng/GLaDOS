"""Shrink an inference copy after a server reports its actual context limit."""

import json
from typing import Any


def reduce_request_context(
    messages: list[dict[str, Any]],
    prompt_tokens: int,
    context_tokens: int,
) -> list[dict[str, Any]]:
    """Keep instructions/current input; remove complete older turns, then excerpt tools.

    This never changes the conversation store. Server token counts guide the
    reduction and leave 512 tokens for the answer; the caller bounds retries.
    """
    if prompt_tokens <= 0 or context_tokens <= 512:
        return messages
    current = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"), 0)
    characters = len(json.dumps(messages, ensure_ascii=False))
    target = max(1, int(characters * (1 - (context_tokens - 512) / prompt_tokens) * 1.3))
    groups: list[list[int]] = []
    group: list[int] = []
    for index, message in enumerate(messages[:current]):
        if message.get("role") == "system":
            continue
        if message.get("role") == "user" and group:
            groups.append(group)
            group = []
        group.append(index)
    if group:
        groups.append(group)
    removed: set[int] = set()
    saved = 0
    for group in groups:
        removed.update(group)
        saved += sum(len(json.dumps(messages[index], ensure_ascii=False)) for index in group)
        if saved >= target:
            break
    result = [message for index, message in enumerate(messages) if index not in removed]
    if saved < target:
        tools = sorted(
            (i for i, m in enumerate(result) if m.get("role") == "tool" and isinstance(m.get("content"), str)),
            key=lambda i: len(result[i]["content"]),
            reverse=True,
        )
        for index in tools:
            content = result[index]["content"]
            notice = "\n[Tool output excerpt shortened to fit model context.]"
            keep = max(512, len(content) - (target - saved) - len(notice))
            if keep + len(notice) >= len(content):
                continue
            result[index] = {**result[index], "content": content[:keep] + notice}
            saved += len(content) - len(result[index]["content"])
            if saved >= target:
                break
    return result
