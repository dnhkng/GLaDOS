"""desktop: Hyprland windows, audio, input, clipboard and the screen (OCR and vision model)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

import requests

from .base import _AgentTool

DESKTOP_ACTIONS = {
    "open_app": "value=command",
    "open_url": "value=URL or file",
    "workspace": "value=number",
    "move_to_workspace": "active window, value=number",
    "focus_window": "value=window class",
    "close_window": "active window",
    "fullscreen": "toggle, active window",
    "windows": "list open windows",
    "active_window": "focused window",
    "volume": "value=40%, +5%, -5% or mute",
    "brightness": "value=50%, +10% or 10%-",
    "media": "value=play-pause, next, previous or stop",
    "notify": "value=text",
    "screenshot": "save to ~/Pictures, returns path",
    "clipboard_get": "read clipboard",
    "clipboard_set": "value=text",
    "lock": "lock the session",
    "read_screen": "OCR the text on screen; value=window for the focused window only",
    "describe_screen": "describe the screen with the vision model; value=question",
    "type_text": "type into the focused window, value=text",
    "key": "press keys, value=e.g. Return, ctrl+c, super+1",
    "move_cursor": "value=x y in screen pixels",
    "click": "value=left, right or middle (needs ydotool)",
    "locate": "find an on-screen element with the vision model, value=what to find; returns x y",
    "click_on": "find an element with the vision model and left-click it, value=what to click",
}


class Desktop(_AgentTool):
    name = "desktop"
    untrusted = True

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        action, value = args.get("action"), str(args.get("value") or "").strip()
        if action not in DESKTOP_ACTIONS:
            raise ValueError("Unknown desktop action")
        needs_value = {"open_app", "open_url", "workspace", "move_to_workspace", "focus_window", "volume"}
        needs_value |= {"brightness", "media", "notify", "clipboard_set", "type_text", "key", "move_cursor"}
        if action in needs_value and not value:
            raise ValueError(f"{action} needs a value")
        # Typing into a terminal runs a command just as surely as run_shell does.
        guarded = action in {"open_app", "type_text"}
        rule = self.guard.blacklisted(value) if guarded else None
        if refusal := self.guard.check(rule and f"command matches {rule}", f"desktop:{action}:{value}", value):
            return refusal
        hypr = ["hyprctl", "dispatch"]
        match action:
            case "open_app":
                return self.execute([*hypr, "exec", value], 5)
            case "open_url":
                return self.execute([*hypr, "exec", "xdg-open " + _quote(value)], 5)
            case "workspace" | "move_to_workspace":
                return self.execute([*hypr, "workspace" if action == "workspace" else "movetoworkspace", value], 5)
            case "focus_window":
                return self.execute([*hypr, "focuswindow", value if ":" in value else f"class:(?i).*{value}.*"], 5)
            case "close_window":
                return self.execute([*hypr, "killactive"], 5)
            case "fullscreen":
                return self.execute([*hypr, "fullscreen", "0"], 5)
            case "windows" | "active_window":
                result = self.execute(
                    ["hyprctl", "clients" if action == "windows" else "activewindow", "-j"], 5, clean=False
                )
                if result["ok"]:
                    data = json.loads(result["stdout"] or "null")
                    windows = data if isinstance(data, list) else [data] if data else []
                    keys = ("class", "title", "workspace")
                    result["stdout"] = self.trim(
                        json.dumps([{k: w.get(k) for k in keys} for w in windows], ensure_ascii=False)
                    )
                return result
            case "volume":
                sink = "@DEFAULT_AUDIO_SINK@"
                if value == "mute":
                    return self.execute(["wpctl", "set-mute", sink, "toggle"], 5)
                amount = value.lstrip("+-")
                step = amount + value[0] if value[0] in "+-" else amount
                return self.execute(["wpctl", "set-volume", "-l", "1.0", sink, step], 5)
            case "brightness":
                return self.execute(["brightnessctl", "set", value], 5)
            case "media":
                return self.execute(["playerctl", value], 5)
            case "notify":
                return self.execute(["notify-send", "GLaDOS", value], 5)
            case "screenshot":
                target = Path("~/Pictures").expanduser() / time.strftime("glados-%Y%m%d-%H%M%S.png")
                target.parent.mkdir(parents=True, exist_ok=True)
                return {**self.execute(["grim", str(target)], 10), "path": str(target)}
            case "clipboard_get":
                return self.execute(["wl-paste", "--no-newline"], 5)
            case "clipboard_set":
                process = subprocess.run(["wl-copy"], input=value, text=True, timeout=5, check=False)
                return {"ok": process.returncode == 0}
            case "lock":
                return self.execute(["loginctl", "lock-session"], 5)
            case "read_screen":
                return self.read_screen(window=value == "window")
            case "describe_screen":
                return self.describe_screen(value or "Describe what is on the screen.")
            case "locate":
                return self.locate(value)
            case "click_on":
                found = self.locate(value)
                if not found.get("found"):
                    return found
                moved = self.execute([*hypr, "movecursor", str(found["x"]), str(found["y"])], 5)
                if not moved["ok"]:
                    return moved
                return {**self.click("left"), "clicked_at": [found["x"], found["y"]]}
            case "type_text":
                return self.execute(["wtype", "--", value], 15)
            case "key":
                *mods, key = [part.strip() for part in value.split("+")]
                mods = [{"super": "logo", "win": "logo", "control": "ctrl"}.get(m.lower(), m.lower()) for m in mods]
                argv = [
                    "wtype",
                    *[a for m in mods for a in ("-M", m)],
                    "-k",
                    key,
                    *[a for m in mods for a in ("-m", m)],
                ]
                return self.execute(argv, 5)
            case "move_cursor":
                x, _, y = value.replace(",", " ").partition(" ")
                return self.execute([*hypr, "movecursor", x.strip(), y.strip()], 5)
            case _:
                return self.click(value or "left")

    def click(self, which: str) -> dict[str, Any]:
        button = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}.get(which)
        if button is None:
            raise ValueError("click value must be left, right or middle")
        if shutil.which("ydotool") is None:
            return {"ok": False, "error": "Clicking needs ydotool (sudo pacman -S ydotool, then run ydotoold)"}
        return self.execute(["ydotool", "click", button], 5)

    def focused_monitor(self) -> dict[str, Any]:
        monitors = json.loads(self.execute(["hyprctl", "monitors", "-j"], 5, clean=False)["stdout"] or "[]")
        return next((m for m in monitors if m.get("focused")), monitors[0] if monitors else {})

    def locate(self, target: str) -> dict[str, Any]:
        """Ask the vision model where ``target`` is; answer in Hyprland's global layout coordinates."""
        if not target:
            raise ValueError("Say what to find")
        monitor = self.focused_monitor()
        scale = float(monitor.get("scale") or 1)
        width, height = round(monitor.get("width", 0) / scale), round(monitor.get("height", 0) / scale)
        # Scale 1 captures in logical pixels, so image coordinates are layout coordinates on this monitor.
        image = self.screenshot(window=False, logical=True)
        answer = self.ask_vision(
            f"This screenshot is {width}x{height} pixels. Find: {target!r}. Reply with only a JSON object giving "
            'the pixel coordinates of its centre, in the form {"x": X, "y": Y}, or {"x": null, "y": null} if it '
            "is not visible.",
            image,
        )
        match = re.search(r"\{[^{}]*\}", answer)
        try:
            point = json.loads(match.group(0)) if match else {}
        except ValueError:
            point = {}
        x, y = point.get("x"), point.get("y")
        if (
            not isinstance(x, int | float)
            or not isinstance(y, int | float)
            or not (0 <= x <= width and 0 <= y <= height)
        ):
            return {
                "ok": False,
                "found": False,
                "error": f"Could not find {target!r} on screen",
                "model_reply": answer[:200],
            }
        return {"ok": True, "found": True, "x": round(monitor.get("x", 0) + x), "y": round(monitor.get("y", 0) + y)}

    def ask_vision(self, prompt: str, image: bytes) -> str:
        if not self.config.vision_model:
            raise ValueError("No vision model configured (agent.vision_model); use read_screen instead")
        response = requests.post(
            self.config.vision_url,
            json={
                "model": self.config.vision_model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image).decode()]}],
            },
            timeout=max(5.0, self.timeout - 5),
        )
        response.raise_for_status()
        return str(response.json().get("message", {}).get("content", ""))

    def screenshot(self, window: bool, logical: bool = False) -> bytes:
        argv = ["grim", "-s", "1"] if logical else ["grim"]
        if window:
            active = json.loads(self.execute(["hyprctl", "activewindow", "-j"], 5, clean=False)["stdout"] or "{}")
            if active.get("at") and active.get("size"):
                (x, y), (w, h) = active["at"], active["size"]
                argv += ["-g", f"{x},{y} {w}x{h}"]
        else:
            monitors = json.loads(self.execute(["hyprctl", "monitors", "-j"], 5, clean=False)["stdout"] or "[]")
            focused = next((m["name"] for m in monitors if m.get("focused")), None)
            if focused:
                argv += ["-o", focused]
        captured = subprocess.run([*argv, "-"], capture_output=True, timeout=10, check=False)
        if captured.returncode != 0 or not captured.stdout:
            raise OSError(captured.stderr.decode(errors="replace").strip() or "Screen capture failed")
        return captured.stdout

    def read_screen(self, window: bool) -> dict[str, Any]:
        image = self.screenshot(window)
        ocr = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", self.config.ocr_languages],
            input=image,
            capture_output=True,
            timeout=30,
            check=False,
        )
        text = "\n".join(line for line in ocr.stdout.decode(errors="replace").splitlines() if line.strip())
        return {"ok": ocr.returncode == 0, "text": self.trim(text), "note": "OCR text; layout and images are lost"}

    def describe_screen(self, question: str) -> dict[str, Any]:
        if not self.config.vision_model:
            return {"ok": False, "error": "No vision model configured (agent.vision_model); use read_screen instead"}
        return {"ok": True, "description": self.trim(self.ask_vision(question, self.screenshot(window=False)))}


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"
