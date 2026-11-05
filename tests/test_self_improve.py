"""Self-improvement: isolated worktree, test gate, owner approval, apply with automatic rollback."""

from collections.abc import Iterator
import json
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from glados.tools import self_improve
from glados.tools.agent import AgentConfig, AgentState


class _Sink:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)

    put_nowait = put


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A small git checkout with uncommitted and untracked work, standing in for GLaDOS's own."""
    root = tmp_path / "repo"
    root.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], cwd=root, check=True)
    (root / "app.py").write_text("VALUE = 1\n")
    subprocess.run([*git, "add", "."], cwd=root, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], cwd=root, check=True)
    (root / "app.py").write_text("VALUE = 2\n")  # uncommitted
    (root / "new.py").write_text("NEW = True\n")  # untracked
    monkeypatch.setattr(self_improve, "REPO", root)
    monkeypatch.setattr(self_improve, "WORK_DIR", tmp_path / "proposals")
    monkeypatch.setattr(self_improve, "_notify_desktop", lambda title, body: None)
    real = Path(__file__).resolve().parents[1]
    before = subprocess.run(["git", "worktree", "list"], cwd=real, capture_output=True, text=True, check=False).stdout
    yield root
    after = subprocess.run(["git", "worktree", "list"], cwd=real, capture_output=True, text=True, check=False).stdout
    assert after == before, "tests must never add worktrees to the real GLaDOS checkout"


def _tool(state: AgentState, latest: str = "hello", **extra: Any) -> self_improve.SelfImprove:
    config = {
        "agent": AgentConfig(enabled=True),
        "agent_state": state,
        "conversation_store": SimpleNamespace(snapshot=lambda: [{"role": "user", "content": latest}]),
        "interaction_state": SimpleNamespace(seconds_since_user=lambda: 5.0),
        **extra,
    }
    return self_improve.SelfImprove(_Sink(), config)


def _worker_edits(new_text: str) -> Any:
    def loop(self: Any, task: str, cancelled: threading.Event, prompt: str = "", steps: int | None = None) -> Any:
        Path(self.config.working_dir, "app.py").write_text(new_text)
        assert str(self_improve.REPO) in str(self.config.protected_paths), "the live checkout is protected"
        return "done", "Changed VALUE.", ["edit_file app.py"]

    return loop


def test_the_worktree_matches_the_running_checkout(repo: Path) -> None:
    tool = _tool(AgentState(audit_path=None))
    worktree = repo.parent / "wt"
    tool.prepare(worktree)
    assert (worktree / "app.py").read_text() == "VALUE = 2\n" and (worktree / "new.py").exists()
    assert (repo / "app.py").read_text() == "VALUE = 2\n", "the live checkout is untouched"


def test_a_passing_change_waits_for_approval_then_applies(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = AgentState(audit_path=None)
    tool = _tool(state)
    monkeypatch.setattr(self_improve, "failures", lambda worktree: ({"tests/x.py::flaky"}, True))
    monkeypatch.setattr(self_improve.DelegateTask, "loop", _worker_edits("VALUE = 3\n"))
    meta = {"id": "abcdef12", "request": "set VALUE to 3", "status": "working", "created": 0}
    tool.save(meta)
    result = tool.build(meta, threading.Event())
    assert result.status == "done" and tool.load("abcdef12")["status"] == "ready"
    assert "+VALUE = 3" in (self_improve.WORK_DIR / "abcdef12" / "change.patch").read_text()
    assert (repo / "app.py").read_text() == "VALUE = 2\n", "nothing changes before approval"
    [pending] = state.approvals()
    assert "abcdef12" in pending["summary"]

    launched: list[list[str]] = []
    monkeypatch.setattr(tool, "execute", lambda argv, *a, **k: launched.append(argv) or {"ok": True})
    assert tool.call({"action": "apply", "id": "abcdef12"})["status"] == "needs_owner_code"
    assert not launched
    state.decide(pending["id"], True)
    assert tool.call({"action": "apply", "id": "abcdef12"})["status"] == "applying"
    assert launched[0][:3] == ["systemd-run", "--user", "--collect"]
    script = self_improve.WORK_DIR / "abcdef12" / "apply.sh"
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0
    assert "git apply -R" in script.read_text(), "the script reverts when GLaDOS does not come back"


def test_new_test_failures_reject_the_proposal(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = iter([(set(), True), ({"tests/test_app.py::test_value"}, True)])
    monkeypatch.setattr(self_improve, "failures", lambda worktree: next(runs))
    monkeypatch.setattr(self_improve.DelegateTask, "loop", _worker_edits("VALUE = 'broken'\n"))
    tool = _tool(AgentState(audit_path=None))
    meta = {"id": "0badc0de", "request": "break it", "status": "working", "created": 0}
    result = tool.build(meta, threading.Event())
    saved = tool.load("0badc0de")
    assert result.status == "error" and saved["status"] == "failed_checks"
    assert saved["new_failures"] == ["tests/test_app.py::test_value"]
    with pytest.raises(ValueError, match="not ready"):
        tool.call({"action": "apply", "id": "0badc0de"})


def test_discard_removes_the_worktree(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(self_improve, "failures", lambda worktree: (set(), True))
    monkeypatch.setattr(self_improve.DelegateTask, "loop", _worker_edits("VALUE = 4\n"))
    tool = _tool(AgentState(audit_path=None))
    meta = {"id": "12345678", "request": "x", "status": "working", "created": 0}
    tool.build(meta, threading.Event())
    assert json.loads(json.dumps(tool.call({"action": "list"})))["proposals"][0]["id"] == "12345678"
    tool.call({"action": "discard", "id": "12345678"})
    assert not (self_improve.WORK_DIR / "12345678").exists()
    listing = subprocess.run(["git", "worktree", "list"], cwd=repo, capture_output=True, text=True, check=False)
    assert "12345678" not in listing.stdout


def test_the_worker_gets_the_owners_exact_words(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def loop(self: Any, task: str, cancelled: threading.Event, prompt: str = "", steps: int | None = None) -> Any:
        seen.append(task)
        assert "Editing this copy IS the proposal" in prompt
        return "done", "", []

    monkeypatch.setattr(self_improve.DelegateTask, "loop", loop)
    monkeypatch.setattr(self_improve, "failures", lambda worktree: (set(), True))
    tool = _tool(AgentState(audit_path=None), latest="Change ton annonce pour « Bonjour »")
    tool.tool_config["task_manager"] = SimpleNamespace(submit=lambda *a, **k: None)
    proposal_id = tool.call({"action": "propose", "request": "Propose an announcement change"})["id"]
    result = tool.build(tool.load(proposal_id), threading.Event())
    assert "Change ton annonce pour « Bonjour »" in seen[0] and result.status == "done"


@pytest.mark.parametrize(
    ("message", "dispatched"),
    [
        ("Améliore ton code : change ton annonce", True),
        ("Change de modèle pour qwen3:8b", True),
        ("Peux-tu modifier ton comportement ?", True),
        ("Ouvre Firefox", False),
        ("Change mon fond d'écran", False),
    ],
)
def test_self_improvement_requests_start_a_proposal_directly(message: str, dispatched: bool) -> None:
    import queue

    from glados.core.llm_processor import LanguageModelProcessor
    from glados.tools.self_improve import self_improve_definition

    processor = LanguageModelProcessor.__new__(LanguageModelProcessor)
    processor.extra_tools = [self_improve_definition]
    processor.tool_calls_queue = queue.Queue()
    processor._conversation_store = SimpleNamespace(append=lambda message: None)
    processor._reply_generation = 3
    turn = SimpleNamespace(autonomy_mode=False, llm_message={"role": "user", "content": message})
    assert processor._dispatch_self_improvement(turn) is dispatched
    if dispatched:
        call = processor.tool_calls_queue.get_nowait()
        assert json.loads(call["function"]["arguments"]) == {"action": "propose", "request": message}


def test_a_change_that_breaks_the_startup_config_is_caught(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(self_improve, "REPO", real)  # for the virtualenv only
    worktree = tmp_path / "wt"
    (worktree / "configs").mkdir(parents=True)
    (worktree / "src").symlink_to(real / "src")
    for name in self_improve.CONFIG_FILES:
        (worktree / name).write_text((real / name).read_text())
    subprocess.run(["git", "init", "-q"], cwd=worktree, check=True)
    assert self_improve.config_problem(worktree) is None
    profile = worktree / "configs/my_glados.yaml"
    # The real failure: a worker wrote read_file's numbered lines back into the profile.
    numbered = "".join(f"{i}\t{line}\n" for i, line in enumerate(profile.read_text().splitlines(), 1))
    profile.write_text(numbered)
    assert self_improve.config_problem(worktree) is not None
