import argparse
import os
from pathlib import Path
import platform
from shutil import which
import subprocess
import sys


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
    4. Detects CUDA availability
    5. Installs the project in editable mode with appropriate dependencies
    6. Downloads and verifies project model files

    The function handles different platform-specific configurations and supports both CUDA and CPU-only installations.

    Notes:
        - Requires UV package manager to be available
        - Assumes project is structured with a standard Python project layout
        - Modifies system environment variables during execution
    """
    parser = argparse.ArgumentParser(description="Set up the project development environment.")
    parser.add_argument("--api", action="store_true", help="Install API dependencies.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Install UV
    install_uv()

    # Create virtual environment
    subprocess.run([*uv_command(), "venv", "--python", "3.12.8"])

    venv_bin = ".venv\\Scripts" if os.name == "nt" else ".venv/bin"

    try:
        has_cuda = subprocess.run(["nvcc", "--version"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        has_cuda = False

    extras = ["cuda"] if has_cuda else ["cpu"]
    if args.api:
        extras.append("api")

    # Install project in editable mode
    env = os.environ.copy()
    env["PATH"] = f"{os.path.abspath(venv_bin)}:{env['PATH']}"
    os.environ["VIRTUAL_ENV"] = os.path.abspath(".venv")
    subprocess.run([*uv_command(), "pip", "install", "-e", f".[{','.join(extras)}]"], env=env)

    # Download and verify model files
    subprocess.run([*uv_command(), "run", "glados", "download"], env=env)


if __name__ == "__main__":
    main()
