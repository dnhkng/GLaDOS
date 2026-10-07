"""Installer planning tests; no downloads, driver changes or package installs."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location("glados_installer", Path(__file__).parents[1] / "scripts" / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_rocm_detection_reads_custom_install_root(tmp_path, monkeypatch):
    info = tmp_path / ".info"
    info.mkdir()
    (info / "version").write_text("7.2.1-12345\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    assert installer.detect_rocm_version() == "7.2.1"


@pytest.mark.parametrize("version", ["7.1", "7.1.0", "7.2", "7.2.0", "7.2.1"])
def test_wheel_matches_rocm_release(version, monkeypatch):
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    url = installer.amd_wheel_url(version)
    release = version[:-2] if version.endswith(".0") else version
    assert f"/rocm-rel-{release}/" in url
    assert url.startswith("https://repo.radeon.com/")
    assert "cp312-cp312" in url


@pytest.mark.parametrize("version", [None, "6.0", "8.0"])
def test_unsupported_rocm_is_rejected(version, monkeypatch):
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="No supported AMD wheel"):
        installer.amd_wheel_url(version)


@pytest.mark.parametrize("system,machine", [("Windows", "AMD64"), ("Linux", "aarch64")])
def test_unsupported_amd_platform_is_rejected(system, machine, monkeypatch):
    monkeypatch.setattr(installer.platform, "system", lambda: system)
    monkeypatch.setattr(installer.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Linux x86_64"):
        installer.amd_wheel_url("7.2.1")


@pytest.mark.parametrize("cuda,rocm,expected", [(True, "7.2.1", "cuda"), (False, "7.2.1", "amd"), (False, None, "cpu")])
def test_auto_backend(cuda, rocm, expected, monkeypatch):
    monkeypatch.setattr(installer, "command_available", lambda command: cuda)
    assert installer.select_backend("auto", rocm) == expected
    assert installer.select_backend("cpu", rocm) == "cpu"


@pytest.mark.parametrize("backend", ["amd", "cuda", "cpu"])
def test_install_uses_one_runtime_and_preserves_it_during_download(backend, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["install.py", "--backend", backend, "--api"])
    monkeypatch.setattr(installer, "detect_rocm_version", lambda: "7.2.1")
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(installer.os, "chdir", lambda path: None)
    monkeypatch.setattr(installer, "install_uv", lambda: None)
    monkeypatch.setattr(installer, "uv_command", lambda: ["uv"])

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(installer.subprocess, "run", run)
    installer.main()
    installs = [call for call in calls if call[:3] == ["uv", "pip", "install"]]
    removal = next(call for call in calls if call[:3] == ["uv", "pip", "uninstall"])
    assert calls.index(removal) < calls.index(installs[0])
    assert ["uv", "run", "--no-sync", "glados", "download"] in calls
    if backend == "amd":
        assert len(installs) == 2
        assert installs[0][-1].startswith("https://repo.radeon.com/")
        assert installs[1][-1] == ".[api]"
        assert any("VAD().ort_sess" in call[-1] for call in calls)
    else:
        assert len(installs) == 1
        assert installs[0][-1] == f".[{backend},api]"


def test_install_failure_prevents_model_download(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["install.py", "--backend", "cpu"])
    monkeypatch.setattr(installer, "detect_rocm_version", lambda: None)
    monkeypatch.setattr(installer.os, "chdir", lambda path: None)
    monkeypatch.setattr(installer, "install_uv", lambda: None)
    monkeypatch.setattr(installer, "uv_command", lambda: ["uv"])
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "install" in argv:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(installer.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        installer.main()
    assert not any("download" in call for call in calls)
