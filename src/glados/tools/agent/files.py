"""File tools: read, write, edit and list, behind the protected-path guard."""

from __future__ import annotations

from typing import Any

from .base import ELIDED_MARKERS, _AgentTool


class ReadFile(_AgentTool):
    name = "read_file"
    untrusted = True

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.path(args.get("path"))
        if refusal := self.protected(path):
            return refusal
        if path.is_dir():
            raise ValueError(f"{path} is a directory; use list_dir")
        lines = path.read_text(errors="replace").splitlines()
        offset = max(0, int(args.get("offset", 1)) - 1)
        count = max(1, min(int(args.get("limit", 200)), 2000))
        chunk = lines[offset : offset + count]
        # Plain text, no line numbers: models copy numbered output back into files.
        return {
            "path": str(path),
            "lines": f"{offset + 1}-{offset + len(chunk)} of {len(lines)}",
            "content": self.trim("\n".join(chunk)),
        }


class WriteFile(_AgentTool):
    name = "write_file"

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        path, content = self.path(args.get("path")), args.get("content")
        if not isinstance(content, str):
            raise TypeError("content must be text")
        if refusal := self.unwritable(path):
            return refusal
        if path.exists() and any(marker in content for marker in ELIDED_MARKERS):
            raise ValueError(
                "content still contains a placeholder from read_file ([hidden] secret or …[truncated]…); writing it "
                "would destroy the real text. Change the file with edit_file instead."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if args.get("append") else "w") as handle:
            handle.write(content)
        if content.startswith("#!") or path.suffix in {".sh", ".py", ".pl", ".rb", ".js"}:
            self.guard.state.written.add(str(path))  # Running it later needs the owner's approval.
        return {"ok": True, "path": str(path), "bytes": len(content.encode()), "appended": bool(args.get("append"))}


class EditFile(_AgentTool):
    name = "edit_file"

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        path, old, new = self.path(args.get("path")), args.get("old_text"), args.get("new_text")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("Give old_text (non-empty) and new_text")
        if refusal := self.unwritable(path):
            return refusal
        text = path.read_text()
        count = text.count(old)
        if count != 1:
            raise ValueError(f"old_text must appear exactly once; found {count} times")
        path.write_text(text.replace(old, new, 1))
        return {"ok": True, "path": str(path)}


class ListDir(_AgentTool):
    name = "list_dir"

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        path = self.path(args.get("path") or ".")
        if refusal := self.protected(path):
            return refusal
        pattern = args.get("pattern") or "*"
        entries = sorted(path.rglob(pattern) if args.get("recursive") else path.glob(pattern))[:300]
        listing = [f"{p.relative_to(path)}{'/' if p.is_dir() else ''}" for p in entries if not p.name.startswith(".")]
        folders = sum(1 for entry in listing if entry.endswith("/"))
        # Small models miscount long lists: give the totals.
        return {
            "path": str(path),
            "folders": folders,
            "files": len(listing) - folders,
            "entries": listing,
            "truncated": len(entries) == 300,
        }
