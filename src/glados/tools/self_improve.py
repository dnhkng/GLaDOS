"""Self-improvement: GLaDOS changes her own code or configuration, on the owner's request only.

The worker edits an isolated git worktree of the running checkout, never the live files. The test
suite and lint run before and after; a proposal that adds failures is rejected. A passing proposal
waits for the owner's approval (console or confirmation code). Applying it patches the checkout and
restarts the service from a separate systemd unit, which reverts the patch and restarts again if
GLaDOS does not come back online.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
from typing import Any
import uuid

from ..autonomy.task_manager import TaskResult
from .agent import AgentConfig, DelegateTask, _AgentTool, _Collect, _definition, _notify_desktop

REPO = Path(__file__).resolve().parents[3]
WORK_DIR = Path("~/.glados/self-improve")
SERVICE = "glados.service"
HEALTH_TIMEOUT_S = 180
PROPOSAL_STEPS = 40
# The configuration the service starts with (see glados.service); a change must keep it loadable.
CONFIG_FILES = ("configs/glados_extended_config.yaml", "configs/my_glados.yaml")

IMPROVER_PROMPT = (
    "You are improving GLaDOS's own source code and configuration, at the owner's request. You work in an "
    "isolated copy of the project at {worktree}: edit files only there, never in {repo}. Editing this copy IS the "
    "proposal: make the requested change directly in its files. Nothing you do reaches the running GLaDOS; the "
    "owner reviews and approves your change afterwards, outside your job, so do not build any review, proposal or "
    "confirmation mechanism yourself. Read the relevant code "
    "first and follow its style. Make the smallest change that does what was asked; add or update tests for "
    "behaviour changes. What the owner hears or sees day to day is usually configuration, not code: start with "
    "their profile configs/my_glados.yaml (it overrides configs/glados_extended_config.yaml), which holds for "
    "example the spoken startup announcement (announcement), the language, the wake word, the main model "
    "(llm_model) and the agent: section (vision_model, worker_model...). Search with grep -rn before editing, "
    "and change the setting that actually produces what the owner described. For a new Ollama model, check "
    "`ollama list` and run `ollama pull NAME` if it is missing. Check your work with "
    "exactly these commands (never `uv run`, which would rebuild the environment):\n"
    "  tests: {pytest}\n  lint: {ruff}\n"
    "Some tests may already fail before your change (for example ones that depend on free memory); only avoid "
    "adding new failures. When finished, reply without tool calls: what you changed, in which files, and why."
)


def _git(*args: str, cwd: Path | None = None, check: bool = True) -> str:
    # REPO is read at call time (not as a default argument) so it always names the current checkout.
    done = subprocess.run(["git", *args], cwd=cwd or REPO, capture_output=True, text=True, timeout=120, check=False)
    if check and done.returncode != 0:
        raise OSError(f"git {args[0]} failed: {done.stderr.strip()[:300]}")
    return done.stdout


def _commit(worktree: Path, message: str) -> str:
    _git("add", "-A", cwd=worktree)
    _git(
        "-c", "user.name=GLaDOS", "-c", "user.email=glados@localhost",
        "commit", "-q", "--allow-empty", "--no-verify", "-m", message, cwd=worktree,
    )  # fmt: skip
    return _git("rev-parse", "HEAD", cwd=worktree).strip()


def check_commands(worktree: Path) -> tuple[str, str]:
    """Tests and lint for a worktree, with the live virtualenv importing the worktree's sources.

    Both run at idle CPU and I/O priority so the full suite does not lag the owner's desktop.
    """
    python = REPO / ".venv/bin/python"
    idle = "nice -n 19 ionice -c 3"
    pytest = f"PYTHONPATH={worktree}/src {idle} {python} -m pytest -q -p no:cacheprovider -rf"
    return pytest, f"{idle} {REPO / '.venv/bin/ruff'} check src tests"


def failures(worktree: Path) -> tuple[set[str], bool]:
    """The failing test ids, and whether lint passed."""
    pytest, ruff = check_commands(worktree)
    tests = subprocess.run(pytest, shell=True, cwd=worktree, capture_output=True, text=True, timeout=900, check=False)
    failed = set(re.findall(r"^(?:FAILED|ERROR) (\S+)", tests.stdout, flags=re.MULTILINE))
    if tests.returncode not in (0, 1) and not failed:  # Collection errors or a crash, not test failures.
        failed.add("pytest exited with code " + str(tests.returncode))
    lint = subprocess.run(ruff, shell=True, cwd=worktree, capture_output=True, text=True, timeout=300, check=False)
    return failed, lint.returncode == 0


def tidy(worktree: Path) -> None:
    """Apply ruff's safe fixes and formatting to the files the change touched, like the owner would."""
    changed = [line[3:].split(" -> ")[-1] for line in _git("status", "--porcelain", cwd=worktree).splitlines()]
    for name in changed:  # Editors and git expect a final newline; workers often drop it.
        path = worktree / name
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".md", ".txt", ".json", ".js", ".css", ".html"}:
            text = path.read_text(errors="replace")
            if text and not text.endswith("\n"):
                path.write_text(text + "\n")
    files = [name for name in changed if name.endswith(".py")]
    ruff = str(REPO / ".venv/bin/ruff")
    if files and Path(ruff).exists():
        for argv in ([ruff, "check", "--fix", "-q", *files], [ruff, "format", "-q", *files]):
            subprocess.run(argv, cwd=worktree, capture_output=True, timeout=120, check=False)


def config_problem(worktree: Path) -> str | None:
    """Why the worktree's YAML files or startup configuration would not load, or None."""
    script = (
        "import pathlib, sys, yaml\n"
        "from glados.core.engine import GladosConfig\n"
        "for name in sys.argv[1:]:\n"
        "    yaml.safe_load(pathlib.Path(name).read_text())\n"
        f"GladosConfig.from_yaml([n for n in {list(CONFIG_FILES)!r} if pathlib.Path(n).exists()])\n"
    )
    changed = _git("status", "--porcelain", cwd=worktree).splitlines()
    yaml_files = [
        line[3:] for line in changed if line[3:].endswith((".yaml", ".yml")) and (worktree / line[3:]).exists()
    ]
    if not any((worktree / name).exists() for name in CONFIG_FILES):
        return None
    done = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", script, *yaml_files],
        cwd=worktree,
        env={**os.environ, "PYTHONPATH": str(worktree / "src")},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if done.returncode == 0:
        return None
    return (done.stderr.strip().splitlines() or ["configuration check failed"])[-1][:300]


class SelfImprove(_AgentTool):
    name = "self_improve"

    def proposals_dir(self) -> Path:
        return WORK_DIR.expanduser()

    def meta_path(self, proposal_id: str) -> Path:
        return self.proposals_dir() / proposal_id / "proposal.json"

    def load(self, proposal_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{8}", proposal_id or ""):
            raise ValueError("Give a proposal id (see action list)")
        try:
            return dict(json.loads(self.meta_path(proposal_id).read_text()))
        except OSError:
            raise ValueError(f"No proposal {proposal_id}") from None

    def save(self, meta: dict[str, Any]) -> None:
        path = self.meta_path(meta["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    def call(self, args: dict[str, Any]) -> dict[str, Any]:
        action = args.get("action", "list")
        if action == "propose":
            return self.propose(" ".join(str(args.get("request") or "").split()))
        if action == "list":
            return {"ok": True, "proposals": self.listing()}
        if action == "show":
            meta = self.load(str(args.get("id") or ""))
            patch = self.proposals_dir() / meta["id"] / "change.patch"
            diff = patch.read_text(errors="replace") if patch.exists() else ""
            return {"ok": True, **meta, "diff": self.trim(diff)}
        if action == "apply":
            return self.apply(str(args.get("id") or ""))
        if action == "discard":
            return self.discard(str(args.get("id") or ""))
        raise ValueError("action must be propose, list, show, apply or discard")

    def listing(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.proposals_dir().glob("*/proposal.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            meta = json.loads(path.read_text())
            status_file = path.parent / "apply.status"
            if status_file.exists():
                meta["status"] = status_file.read_text().strip() or meta["status"]
            rows.append({k: meta.get(k) for k in ("id", "request", "status", "files", "summary", "created")})
        return rows[:20]

    # Proposing ------------------------------------------------------------------------------------

    def propose(self, request: str) -> dict[str, Any]:
        if not request:
            raise ValueError("Describe the change to make")
        if not (REPO / ".git").exists():
            raise OSError(f"{REPO} is not a git checkout; self-improvement needs one")
        tasks = self.tool_config.get("task_manager")
        if tasks is None:
            raise ValueError("Background tasks are unavailable in this profile")
        proposal_id = uuid.uuid4().hex[:8]
        meta = {
            "id": proposal_id,
            "request": request[:1000],
            "owner_words": self.owner_words(),
            "status": "working",
            "created": time.time(),
        }
        self.save(meta)
        cancelled = threading.Event()
        tasks.submit(
            "task_self_" + proposal_id,
            "Self-improvement: " + request[:90],
            lambda: self.build(meta, cancelled),
            group="agent",
            cancelled=cancelled,
        )
        return {
            "ok": True,
            "status": "queued",
            "id": proposal_id,
            "instruction": "Tell the user in one short sentence that you are preparing this change in an isolated "
            "copy, will test it, and will ask for approval before applying anything.",
        }

    def owner_words(self) -> str | None:
        """The owner's latest message, so the worker is not misled by a paraphrase."""
        store = self.tool_config.get("conversation_store")
        if store is None:
            return None
        latest = next(
            (
                m.get("content")
                for m in reversed(store.snapshot())
                if m.get("role") == "user" and isinstance(m.get("content"), str)
            ),
            None,
        )
        return latest[:2000] if latest else None

    def build(self, meta: dict[str, Any], cancelled: threading.Event) -> TaskResult:
        folder = self.proposals_dir() / meta["id"]
        worktree = folder / "worktree"
        try:
            base_commit = self.prepare(worktree)
            meta["base"] = base_commit
            baseline, lint_was_ok = failures(worktree)
            config_was_ok = config_problem(worktree) is None
            pytest, ruff = check_commands(worktree)
            prompt = IMPROVER_PROMPT.format(worktree=worktree, repo=REPO, pytest=pytest, ruff=ruff)
            config: AgentConfig = self.config.model_copy(
                update={
                    "working_dir": str(worktree),
                    # The live checkout is off-limits to the worker's file tools.
                    "protected_paths": [*self.config.protected_paths, str(REPO)],
                }
            )
            worker = DelegateTask(_Collect(), {**self.tool_config, "agent": config})
            task = meta["request"]
            if meta.get("owner_words") and meta["owner_words"] != task:
                task += f"\n\nThe owner's exact words (they win if the summary above differs): {meta['owner_words']}"
            status, summary, actions = worker.loop(task, cancelled, prompt=prompt, steps=PROPOSAL_STEPS)
            if status == "cancelled":
                return self.finish(meta, "cancelled", "Self-improvement cancelled")
            changed = _git("status", "--porcelain", cwd=worktree).strip()
            if not changed:
                return self.finish(meta, "no_change", summary or "The worker made no change.", actions=actions)
            tidy(worktree)
            broken_config = config_problem(worktree) if config_was_ok else None
            after, lint_ok = failures(worktree)
            meta["new_failures"] = sorted(after - baseline)
            meta["lint_ok"] = lint_ok or not lint_was_ok  # Only lint errors this change adds count against it.
            lint_ok = meta["lint_ok"]
            commit = _commit(worktree, "Self-improvement: " + meta["request"][:60])
            patch = _git("diff", "--binary", base_commit, commit, cwd=worktree)
            (folder / "change.patch").write_text(patch)
            stat = _git("diff", "--stat", base_commit, commit, cwd=worktree).strip().splitlines()
            meta["files"] = stat[-1].strip() if stat else ""
            meta["config_error"] = broken_config
            if meta["new_failures"] or not lint_ok or broken_config:
                problems = (
                    f"the startup configuration no longer loads ({broken_config})"
                    if broken_config
                    else ", ".join(meta["new_failures"][:5]) or "lint errors"
                )
                return self.finish(meta, "failed_checks", f"{summary}\n\nRejected: {problems}", actions=actions)
            approval = self.guard.state.request(
                f"self_improve:apply:{meta['id']}", f"Apply self-improvement {meta['id']}: {meta['request'][:140]}"
            )
            meta["approval_id"] = approval
            return self.finish(meta, "ready", summary, actions=actions)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return self.finish(meta, "error", str(exc))

    def prepare(self, worktree: Path) -> str:
        """A worktree holding the checkout exactly as it runs now, uncommitted work included."""
        snapshot = _git("stash", "create").strip() or "HEAD"
        _git("worktree", "add", "--detach", str(worktree), snapshot)
        untracked = _git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
        for name in filter(None, untracked):
            target = worktree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        return _commit(worktree, "Snapshot of the running checkout")

    def finish(self, meta: dict[str, Any], status: str, summary: str, actions: list[str] | None = None) -> TaskResult:
        meta.update(status=status, summary=summary[:3000], steps=(actions or [])[-40:], finished=time.time())
        self.save(meta)
        outcome = {
            "ready": "ready for approval",
            "failed_checks": "rejected by tests or lint",
            "no_change": "made no change",
            "cancelled": "cancelled",
            "error": "failed",
        }[status]
        if status != "cancelled":
            _notify_desktop(f"GLaDOS: self-improvement {outcome}", meta["request"][:120])
        note = (
            " Ask the owner to approve it in the console's Agent panel (or say the confirmation code), then call "
            "self_improve apply with this id."
            if status == "ready"
            else ""
        )
        return TaskResult(
            "done" if status in {"ready", "no_change"} else "cancelled" if status == "cancelled" else "error",
            f"Self-improvement {meta['id']} {outcome}: {meta['request'][:100]}.{note}",
            notify_user=status != "cancelled",
            report=json.dumps({k: meta.get(k) for k in ("id", "status", "files", "new_failures", "summary")}),
            importance=0.8,
            update_priority="important",
        )

    # Applying -------------------------------------------------------------------------------------

    def apply(self, proposal_id: str) -> dict[str, Any]:
        meta = self.load(proposal_id)
        if meta.get("status") != "ready":
            raise ValueError(f"Proposal {proposal_id} is {meta.get('status')}, not ready to apply")
        if refusal := self.guard.check(
            "changing GLaDOS's own code", f"self_improve:apply:{proposal_id}", f"Apply self-improvement {proposal_id}"
        ):
            return refusal
        folder = self.proposals_dir() / proposal_id
        patch = folder / "change.patch"
        check = subprocess.run(
            ["git", "apply", "--check", str(patch)], cwd=REPO, capture_output=True, text=True, timeout=60, check=False
        )
        if check.returncode != 0:
            return {"ok": False, "error": "The code changed since this proposal: " + check.stderr.strip()[:300]}
        script = folder / "apply.sh"
        script.write_text(self.apply_script(proposal_id, patch, folder / "apply.status"))
        launched = self.execute(
            ["systemd-run", "--user", "--collect", f"--unit=glados-apply-{proposal_id}", "/bin/bash", str(script)], 10
        )
        if not launched["ok"]:
            return launched
        meta["status"] = "applying"
        self.save(meta)
        return {
            "ok": True,
            "status": "applying",
            "instruction": "Tell the user the change is being applied and that you will restart in a few seconds; "
            "if you do not come back online, the change is reverted automatically.",
        }

    def apply_script(self, proposal_id: str, patch: Path, status: Path) -> str:
        url = self.config.console_url.rstrip("/") + "/api/state"
        return f"""#!/bin/bash
# Apply self-improvement {proposal_id}; revert it if GLaDOS does not come back online.
cd {_sh(str(REPO))} || exit 1
sleep 5  # Let GLaDOS finish telling the owner.
git apply {_sh(str(patch))} || {{ echo apply_failed > {_sh(str(status))}; exit 1; }}
systemctl --user restart {SERVICE}
for _ in $(seq 1 {HEALTH_TIMEOUT_S // 3}); do
  sleep 3
  if [ "$(curl -sk -o /dev/null -w '%{{http_code}}' {_sh(url)})" = 200 ]; then
    echo applied > {_sh(str(status))}
    notify-send -a GLaDOS "GLaDOS: self-improvement applied" {_sh(proposal_id)}
    exit 0
  fi
done
git apply -R {_sh(str(patch))}
systemctl --user restart {SERVICE}
echo rolled_back > {_sh(str(status))}
notify-send -a GLaDOS -u critical "GLaDOS: self-improvement reverted" \\
  "GLaDOS did not restart; {proposal_id} was undone."
"""

    def discard(self, proposal_id: str) -> dict[str, Any]:
        meta = self.load(proposal_id)
        if meta.get("status") == "applying":
            raise ValueError("This proposal is being applied")
        folder = self.proposals_dir() / proposal_id
        worktree = folder / "worktree"
        if worktree.exists():
            _git("worktree", "remove", "--force", str(worktree), check=False)
        _git("worktree", "prune", check=False)
        shutil.rmtree(folder, ignore_errors=True)
        return {"ok": True, "discarded": proposal_id}


def _sh(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


self_improve_definition = _definition(
    "self_improve",
    "Change your own code or configuration (features, behaviour, models) when the owner asks. propose: a worker "
    "edits an isolated copy, then tests and lint run; a passing change waits for the owner's approval. list/show: "
    "proposals and their diff. apply: install an approved proposal and restart (reverted automatically if you do "
    "not come back). discard: drop one. Only on the owner's explicit request.",
    {
        "action": {"type": "string", "enum": ["propose", "list", "show", "apply", "discard"]},
        "request": {"type": "string", "description": "propose: the change the owner asked for, in full"},
        "id": {"type": "string"},
    },
    ["action"],
)
