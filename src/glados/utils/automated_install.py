import asyncio
import os
import subprocess
import sys
from pathlib import Path
import yaml
from .resources import get_package_root

def check_ollama_installed() -> bool:
    """Check if Ollama is installed by running 'ollama --version'."""
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ollama_windows():
    """Download and install Ollama on Windows."""
    download_url = "https://ollama.com/download/OllamaSetup.exe"
    installer_path = Path("OllamaSetup.exe")
    
    try:
        subprocess.run(["curl", "-L", download_url, "-o", str(installer_path)], check=True)
        subprocess.run([str(installer_path)], check=True)
        print("Ollama installed successfully on Windows")
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to install Ollama on Windows")
    finally:
        installer_path.unlink(missing_ok=True)

def install_ollama_linux():
    """Install Ollama on Linux using the official script."""
    try:
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
        print("Ollama installed successfully on Linux")
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to install Ollama on Linux")

def install_ollama_macos():
    """Install Ollama on macOS using Homebrew or direct download."""
    try:
        # Check if Homebrew is installed
        if subprocess.run(["which", "brew"], capture_output=True, text=True).returncode == 0:
            subprocess.run(["brew", "install", "ollama"], check=True)
            print("Ollama installed successfully on macOS via Homebrew")
        else:
            # Fallback to direct download
            download_url = "https://ollama.com/download/Ollama-darwin.zip"
            installer_path = Path("Ollama-darwin.zip")
            app_path = Path("/Applications/Ollama.app")
            
            subprocess.run(["curl", "-L", download_url, "-o", str(installer_path)], check=True)
            subprocess.run(["unzip", str(installer_path), "-d", "/Applications"], check=True)
            print("Ollama installed successfully on macOS to /Applications")
            installer_path.unlink(missing_ok=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to install Ollama on macOS")

def main():
    # We have to import this here to avoid circular imports
    # Meaning that python will try to import cli.py before this function is called
    from ..cli import download_models, models_valid

    if check_ollama_installed():
        print("Ollama is already installed")
    else:
        print("Ollama is not installed")
        if sys.platform == "win32":
            print("Installing Ollama on Windows...")
            install_ollama_windows()
        elif sys.platform == "linux":
            print("Installing Ollama on Linux...")
            install_ollama_linux()
        elif sys.platform == "darwin":
            print("Installing Ollama on macOS...")
            install_ollama_macos()
        else:
            raise RuntimeError("Unsupported platform. This script supports Windows, Linux, and macOS.")
        print("Ollama installation complete")

    # Ensure those configuration files exist in the current working directory
    default_configuration_files = [
        "configs/glados_config.yaml",
        "models/ASR/parakeet-tdt_ctc-110m_model_config.yaml",
        "models/ASR/parakeet-tdt-0.6b-v2_model_config.yaml",
        "models/TTS/glados.json",
        "models/TTS/idx_to_token.pkl",
        "models/TTS/lang_phoneme_dict_old.pkl",
        "models/TTS/lang_phoneme_dict.pkl",
        "models/TTS/phoneme_to_id.pkl",
        "models/TTS/token_to_idx.pkl",
        "data/0.wav"
    ]
    for file in default_configuration_files:
        relative_path = Path(os.getenv("PYAPP_RELATIVE_DIR")) / file
        package_path = get_package_root() / "default_configs" / Path(file).name

        # Ensure the parent directory exists
        relative_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file from the package to the current working directory if it doesn't exist
        if not relative_path.exists():
            with package_path.open("rb") as src, relative_path.open("wb") as dst:
                print(f"Creating default config file {file}")
                dst.write(src.read())
        else:
            print(f"Config file {file} already exists.")

    if not models_valid():
        print("Some model files are invalid or missing, downloading...")
        asyncio.run(download_models())

    with (Path(os.getenv("PYAPP_RELATIVE_DIR")) / "configs/glados_config.yaml").open() as f:
        config = yaml.safe_load(f)
    
    llm_model = config.get("Glados", {}).get("llm_model")
    if not llm_model:
        raise ValueError("No 'llm_model' specified in YAML file")
    
    print(f"Pulling model: {llm_model}")
    try:
        subprocess.run(["ollama", "pull", llm_model], check=True)
        print(f"Model {llm_model} pulled successfully")
    except subprocess.CalledProcessError:
        raise RuntimeError(f"Failed to pull model: {llm_model}")

if __name__ == "__main__":
    main("configs/glados_config.yaml")
