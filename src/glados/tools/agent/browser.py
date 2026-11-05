"""browser: the agent's own Chromium, driven over the DevTools protocol."""

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
from websockets.sync.client import ClientConnection, connect

from .base import _AgentTool
from .desktop import Desktop

BROWSER_ACTIONS = {
    "open": "value=URL",
    "read": "text of the current page",
    "links": "visible links with their text",
    "click": "value=visible text of a link/button, or a CSS selector",
    "type": "selector=CSS selector or field label/placeholder, value=text",
    "press": "value=key, e.g. Enter, Tab, Escape",
    "back": "previous page",
    "look": "describe the page with the vision model, value=question",
}
# Runs in the page: find an element by CSS selector, else by its visible text, label or placeholder.
_FIND_JS = """(query) => {
  let el = null;
  try { el = document.querySelector(query); } catch (e) {}
  if (!el) {
    const q = query.trim().toLowerCase();
    const candidates = [...document.querySelectorAll(
      'a,button,input,textarea,select,[role=button],[role=link],label,summary')];
    const visible = e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
    const text = e => (e.innerText || e.value || e.getAttribute('aria-label') || e.placeholder || e.title || '')
      .trim().toLowerCase();
    el = candidates.find(e => visible(e) && text(e) === q) || candidates.find(e => visible(e) && text(e).includes(q));
    if (el && el.tagName === 'LABEL' && el.control) el = el.control;
  }
  return el;
}"""


class Browser(_AgentTool):
    """A dedicated Chromium driven over the DevTools protocol; its own profile, visible on screen."""

    name = "browser"
    untrusted = True

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        action, value = args.get("action"), str(args.get("value") or "").strip()
        if action not in BROWSER_ACTIONS:
            raise ValueError("Unknown browser action")
        if action in {"open", "click", "type", "press"} and not value:
            raise ValueError(f"{action} needs a value")
        page = self.page()
        with connect(page["webSocketDebuggerUrl"], max_size=None, open_timeout=10) as socket:
            cdp = _Cdp(socket)
            match action:
                case "open":
                    url = value if re.match(r"^[a-z]+:", value) else "https://" + value
                    cdp.call("Page.navigate", {"url": url})
                    return self.settled(cdp)
                case "back":
                    cdp.evaluate("history.back()")
                    return self.settled(cdp)
                case "read":
                    text = cdp.evaluate("document.body ? document.body.innerText : ''")
                    return {"ok": True, **self.where(cdp), "text": self.trim(str(text))}
                case "links":
                    links = cdp.evaluate(
                        "[...document.querySelectorAll('a[href]')].filter(a => a.getBoundingClientRect().height > 0)"
                        ".slice(0, 80).map(a => [a.innerText.trim().slice(0, 80), a.href])"
                    )
                    lines = [
                        f"{text or '(no text)'} -> {href}" for text, href in (links if isinstance(links, list) else [])
                    ]
                    return {"ok": True, **self.where(cdp), "links": self.trim("\n".join(lines))}
                case "click":
                    clicked = cdp.evaluate(
                        f"(() => {{ const el = ({_FIND_JS})({json.dumps(value)}); if (!el) return null; "
                        "el.scrollIntoView({block: 'center'}); el.click(); "
                        "return (el.innerText || el.value || el.tagName).trim().slice(0, 80); })()"
                    )
                    if clicked is None:
                        return {"ok": False, "error": f"Nothing matches {value!r} on this page"}
                    return {**self.settled(cdp), "clicked": clicked}
                case "type":
                    selector = str(args.get("selector") or "").strip()
                    if not selector:
                        raise ValueError("type needs selector (the field) and value (the text)")
                    field = cdp.evaluate(
                        f"(() => {{ const el = ({_FIND_JS})({json.dumps(selector)}); if (!el) return null; "
                        "el.scrollIntoView({block: 'center'}); el.focus(); if ('select' in el) el.select(); "
                        "return el.type || el.tagName; })()"
                    )
                    if field is None:
                        return {"ok": False, "error": f"No field matches {selector!r}"}
                    if field == "password" and (
                        refusal := self.guard.check("typing a password", f"browser:password:{selector}", selector)
                    ):
                        return refusal
                    cdp.call("Input.insertText", {"text": value})
                    return {"ok": True, "typed_into": field}
                case "press":
                    for kind in ("keyDown", "keyUp"):
                        cdp.call("Input.dispatchKeyEvent", {"type": kind, **_key(value)})
                    return self.settled(cdp)
                case _:
                    shot = cdp.call("Page.captureScreenshot", {"format": "png"})["data"]
                    question = value or "Describe this web page: what it shows and what can be done on it."
                    answer = Desktop(self.llm_queue, self.tool_config).ask_vision(question, base64.b64decode(shot))
                    return {"ok": True, **self.where(cdp), "description": self.trim(answer)}

    def page(self) -> dict[str, Any]:
        """The browser's first tab, launching the browser when it is not running yet."""
        endpoint = f"http://127.0.0.1:{self.config.browser_port}/json"
        for attempt in range(40):
            try:
                pages = [t for t in requests.get(endpoint, timeout=2).json() if t.get("type") == "page"]
                if pages:
                    return dict(pages[0])
                requests.put(endpoint + "/new?about:blank", timeout=2)
            except requests.RequestException:
                if attempt == 0:
                    self.launch()
            time.sleep(0.25)
        raise OSError("The browser did not start")

    def launch(self) -> None:
        binary = next((b for b in ("chromium", "google-chrome", "google-chrome-stable") if shutil.which(b)), None)
        if binary is None:
            raise OSError("No Chromium-based browser is installed")
        subprocess.Popen(
            [
                binary,
                f"--remote-debugging-port={self.config.browser_port}",
                "--remote-allow-origins=http://127.0.0.1",
                f"--user-data-dir={Path(self.config.browser_profile).expanduser()}",
                "--no-first-run",
                "--no-default-browser-check",
                "about:blank",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def where(self, cdp: _Cdp) -> dict[str, Any]:
        return {"url": cdp.evaluate("location.href"), "title": cdp.evaluate("document.title")}

    def settled(self, cdp: _Cdp) -> dict[str, Any]:
        """Wait for the page to finish loading (bounded), then say where we are."""
        deadline = time.monotonic() + min(20.0, max(3.0, self.timeout - 5))
        time.sleep(0.3)
        while time.monotonic() < deadline and cdp.evaluate("document.readyState") != "complete":
            time.sleep(0.25)
        return {"ok": True, **self.where(cdp)}


class _Cdp:
    """Minimal synchronous DevTools protocol client on one page socket."""

    def __init__(self, socket: ClientConnection) -> None:
        self.socket = socket
        self.next_id = 0

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        self.socket.send(json.dumps({"id": self.next_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv(timeout=30))
            if message.get("id") == self.next_id:
                if "error" in message:
                    raise OSError(f"{method}: {message['error'].get('message')}")
                return dict(message.get("result") or {})

    def evaluate(self, expression: str) -> object:
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in result:
            raise OSError("Page script failed: " + str(result["exceptionDetails"].get("text", ""))[:200])
        return result.get("result", {}).get("value")


def _key(name: str) -> dict[str, Any]:
    codes = {"Enter": 13, "Tab": 9, "Escape": 27, "Backspace": 8, "ArrowDown": 40, "ArrowUp": 38, "Space": 32}
    key = {"Return": "Enter", "Esc": "Escape"}.get(name, name)
    code = codes.get(key, ord(key[0].upper()) if len(key) == 1 else 0)
    event: dict[str, Any] = {"key": key, "windowsVirtualKeyCode": code, "nativeVirtualKeyCode": code}
    if key == "Enter":
        event["text"] = "\r"
    return event
