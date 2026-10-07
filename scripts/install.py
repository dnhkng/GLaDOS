import argparse
import os
from pathlib import Path
import platform
import re
from shutil import which
import subprocess
import sys

AMD_WHEELS = {
    "7.1": "onnxruntime_migraphx-1.23.1",
    "7.2": "onnxruntime_migraphx-1.23.2",
    "7.2.1": "onnxruntime_migraphx-1.23.2",
}
ONNX_PACKAGES = ("onnxruntime", "onnxruntime-gpu", "onnxruntime-rocm", "onnxruntime-migraphx")


def detect_rocm_version() -> str | None:
    """Read the installed ROCm version without changing system drivers."""
    root = Path(os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm")
    for filename in ("version", "version-dev"):
        path = root / ".info" / filename
        if path.is_file():
            match = re.match(r"(\d+\.\d+(?:\.\d+)?)", path.read_text().strip())
            if match:
                return match.group(1)
    return None


def command_available(command: str) -> bool:
    """Check whether an installed GPU utility can run."""
    try:
        return subprocess.run([command, "--version"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        return False


def select_backend(requested: str, rocm_version: str | None) -> str:
    if requested != "auto":
        return requested
    if command_available("nvcc") or command_available("nvidia-smi"):
        return "cuda"
    return "amd" if rocm_version else "cpu"


def amd_wheel_url(version: str | None) -> str:
    """Select an official Python 3.12 wheel matched to a supported ROCm release."""
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise ValueError("AMD installation currently supports Linux x86_64 only.")
    if version and version.endswith(".0"):
        version = version[:-2]
    if version not in AMD_WHEELS:
        supported = ", ".join(AMD_WHEELS)
        raise ValueError(
            f"No supported AMD wheel for ROCm {version or 'unknown'}. "
            f"Install ROCm first, then use --rocm-version with one of: {supported}. "
            "Use --backend cpu to install without AMD acceleration."
        )
    filename = AMD_WHEELS[version] + "-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
    return f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{version}/{filename}"


def is_uv_installed() -> bool:
    """
    Check if the UV tool is installed on the system.

    Returns:
        bool: True if the 'uv' command is available in the system path, False otherwise.
    """
    return which("uv") is not None


def uv_command() -> list[str]:
    """
    Resolve the best way to invoke 'uv' on this system.

    Prefers the 'uv' executable on PATH, but falls back to the uv Python module
    ("python -m uv"). A Windows pip user-install can place uv on a directory that
    is not on PATH, which would otherwise crash every later 'uv' call with
    FileNotFoundError (WinError 2).

    Returns:
        list[str]: The argv to invoke uv, e.g. ['uv'] or [sys.executable, '-m', 'uv'].
    """
    if which("uv") is not None:
        return ["uv"]
    return [sys.executable, "-m", "uv"]


def install_uv() -> None:
    """
    Install the UV package management tool across different platforms.

    This function checks if UV is already installed. If not, it performs the installation:
    - On Windows, it uses pip to install UV and then updates it
    - On other platforms, it uses a curl-based installation script from Astral.sh

    Raises:
        subprocess.CalledProcessError: If the pip-based installation fails
    """
    if is_uv_installed():
        print("UV is already installed")
        return

    print("Installing UV...")
    if platform.system() == "Windows":
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    elif platform.system() == "PotatOS":
        raise Exception("Oh no. Not again.")
    else:
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)

    try:
        subprocess.run([*uv_command(), "self", "update"])
    except FileNotFoundError:
        print("WARNING: 'uv' executable is not available on PATH; subsequent commands will use 'python -m uv'.")


def main() -> None:
    """
    Set up the project development environment by installing UV, creating a virtual environment,
    and preparing the project for development.

    This function performs the following steps:
    1. Changes the current working directory to the project root
    2. Installs the UV package management tool
    3. Creates a Python 3.12.8 virtual environment
    4. Selects CPU, CUDA or an installed AMD ROCm backend
    5. Installs the project in editable mode with appropriate dependencies
    6. Downloads and verifies project model files

    AMD uses an official MIGraphX wheel matched to the installed ROCm release.

    Notes:
        - Requires UV package manager to be available
        - Assumes project is structured with a standard Python project layout
        - Modifies system environment variables during execution
    """
    parser = argparse.ArgumentParser(description="Set up the project development environment.")
    parser.add_argument("--api", action="store_true", help="Install API dependencies.")
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda", "amd"), default="auto")
    parser.add_argument("--rocm-version", help="Installed ROCm release, overriding automatic detection (AMD only).")
    args = parser.parse_args()

    rocm_version = args.rocm_version or detect_rocm_version()
    backend = select_backend(args.backend, rocm_version)
    if args.rocm_version and backend != "amd":
        parser.error("--rocm-version requires --backend amd or automatic AMD selection.")
    try:
        wheel = amd_wheel_url(rocm_version) if backend == "amd" else None
    except ValueError as error:
        parser.error(str(error))
    print(f"Installing GLaDOS with the {backend} backend.")

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Install UV
    install_uv()

    # Create virtual environment
    subprocess.run([*uv_command(), "venv", "--allow-existing", "--python", "3.12.8"], check=True)

    venv_bin = ".venv\\Scripts" if os.name == "nt" else ".venv/bin"

    extras = [] if backend == "amd" else [backend]
    if args.api:
        extras.append("api")

    # Install project in editable mode
    env = os.environ.copy()
    env["PATH"] = f"{os.path.abspath(venv_bin)}:{env['PATH']}"
    env["VIRTUAL_ENV"] = os.path.abspath(".venv")
    # These distributions all provide the same Python module. Remove the old
    # backend before installing another one to avoid overlapping package files.
    subprocess.run([*uv_command(), "pip", "uninstall", *ONNX_PACKAGES], env=env, check=True)
    if wheel:
        subprocess.run([*uv_command(), "pip", "install", wheel], env=env, check=True)
    project = f".[{','.join(extras)}]" if extras else "."
    subprocess.run([*uv_command(), "pip", "install", "-e", project], env=env, check=True)

    venv_python = str(Path(venv_bin) / ("python.exe" if os.name == "nt" else "python"))
    probe = "import onnxruntime as ort; p = ort.get_available_providers(); print('Available ONNX providers:', p)"
    if backend == "amd":
        probe += "; assert 'MIGraphXExecutionProvider' in p, 'AMD runtime is missing MIGraphX; check ROCm installation'"
    subprocess.run([venv_python, "-c", probe], env=env, check=True)

    # Download and verify model files
    # Do not let uv sync replace the separately installed vendor runtime.
    subprocess.run([*uv_command(), "run", "--no-sync", "glados", "download"], env=env, check=True)
    if backend == "amd":
        probe = (
            "from glados.audio_io.vad import VAD; "
            "p = VAD().ort_sess.get_providers(); print('VAD session providers:', p); "
            "assert 'MIGraphXExecutionProvider' in p, 'AMD session fell back to CPU; check ROCm libraries'"
        )
        subprocess.run([venv_python, "-c", probe], env=env, check=True)
        print("AMD setup complete. Launch with: uv run --no-sync glados")


if __name__ == "__main__":
    main()
