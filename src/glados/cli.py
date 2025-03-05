import argparse
import hashlib
from pathlib import Path
import asyncio

import requests
import sounddevice as sd  # type: ignore

from .engine import Glados, GladosConfig
from .TTS import tts_glados
from .utils import spoken_text_converter as stc
from .twitch import start_twitch_bot

DEFAULT_CONFIG = Path("configs/glados_config.yaml")

MODEL_CHECKSUMS = {
    "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx": "313705ff6f897696ddbe0d92b5ffadad7429a47d2ddeef370e6f59248b1e8fb5",
    "models/ASR/silero_vad_v5.onnx": "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396",
    "models/TTS/glados.onnx": "17ea16dd18e1bac343090b8589042b4052f1e5456d42cad8842a4f110de25095",
    "models/TTS/kokoro-v1.0.fp16.onnx": "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    "models/TTS/kokoro-voices-v1.0.bin": "c5adf5cc911e03b76fa5025c1c225b141310d0c4a721d6ed6e96e73309d0fd88",
    "models/TTS/phomenizer_en.onnx": "b64dbbeca8b350927a0b6ca5c4642e0230173034abd0b5bb72c07680d700c5a0",
}

MODEL_URLS = {
    "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/nemo-parakeet_tdt_ctc_110m.onnx",
    "models/ASR/silero_vad_v5.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/silero_vad_v5.onnx",
    "models/TTS/glados.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/glados.onnx",
    "models/TTS/kokoro-v1.0.fp16.onnx": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-v1.0.fp16.onnx",
    "models/TTS/kokoro-voices-v1.0.bin": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-voices-v1.0.bin",
    "models/TTS/phomenizer_en.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/phomenizer_en.onnx",
}

assert MODEL_CHECKSUMS.keys() == MODEL_URLS.keys()


def verify_checksums() -> dict[str, bool]:
    """
    Verify the integrity of model files by comparing their SHA-256 checksums against expected values.

    This function checks each model file specified in MODEL_CHECKSUMS to ensure it exists
    and has the correct checksum. Files that are missing or have incorrect checksums are
    marked as invalid.

    Parameters:
        None

    Returns:
        dict[str, bool]: A dictionary where keys are model file paths and values are
                         boolean indicators of checksum validity (True if valid, False if
                         missing or checksum mismatch)

    Example:
        >>> verify_checksums()
        {
            'models/tts_model.pth': True,
            'models/asr_model.bin': False
        }
    """
    results = {}
    for path, expected in MODEL_CHECKSUMS.items():
        model_path = Path(path)
        if not model_path.exists():
            results[path] = False
            continue

        sha = hashlib.sha256()
        sha.update(model_path.read_bytes())
        results[path] = sha.hexdigest() == expected

    return results


def download_models() -> None:
    """Download model files required for GLaDOS with robust error handling and progress tracking.

    This function downloads the required model files from predefined URLs if they are missing
    or have invalid checksums. It includes several reliability features:

    Features:
        - Progress tracking with percentage display
        - Temporary file usage to prevent partial downloads
        - SHA256 checksum verification during download
        - Retry logic with exponential backoff (max 3 retries)
        - Network timeout handling (30 seconds)
        - Automatic cleanup of temporary files on failure
        - Creation of parent directories if they don't exist

    The function downloads the following models:
        - ASR model: nemo-parakeet_tdt_ctc_110m.onnx
        - VAD model: silero_vad.onnx
        - TTS model: glados.onnx
        - Phonemizer model: phomenizer_en.onnx

    Raises:
        requests.RequestException: If network errors occur during download
        IOError: If file system operations fail
        ValueError: If downloaded file has incorrect checksum
        SystemExit: If any model download fails after max retries

    Example:
        >>> from glados.cli import download_models
        >>> download_models()  # Downloads all required models with progress tracking
    """
    import sys
    from time import sleep
    from typing import BinaryIO

    def calculate_sha256(file: BinaryIO) -> str:
        """Calculate SHA256 hash of a file-like object."""
        sha = hashlib.sha256()
        file.seek(0)
        for chunk in iter(lambda: file.read(8192), b""):
            sha.update(chunk)
        file.seek(0)
        return sha.hexdigest()

    def download_with_progress(url: str, path: Path, expected_hash: str, max_retries: int = 3) -> None:
        """Download a file with progress tracking and retry logic."""
        retry_count = 0
        temp_path = path.with_suffix(path.suffix + ".tmp")

        while retry_count < max_retries:
            try:
                # Create parent directories if they don't exist
                path.parent.mkdir(parents=True, exist_ok=True)

                # Start download with stream enabled
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                # Open temporary file for writing
                with open(temp_path, "wb") as f:
                    if total_size == 0:
                        print(f"\nDownloading {path.name} (size unknown)")
                    else:
                        print(f"\nDownloading {path.name} ({total_size / 1024 / 1024:.1f} MB)")

                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                sys.stdout.write(f"\rProgress: {progress:.1f}%")
                                sys.stdout.flush()

                # Verify checksum
                with open(temp_path, "rb") as f:
                    if calculate_sha256(f) != expected_hash:
                        raise ValueError("Downloaded file has incorrect checksum")

                # Rename temporary file to final path
                if path.exists():
                    path.unlink()
                temp_path.rename(path)
                print(f"\nSuccessfully downloaded {path.name}")
                return

            except (OSError, requests.RequestException, ValueError) as e:
                retry_count += 1
                if temp_path.exists():
                    temp_path.unlink()

                if retry_count < max_retries:
                    wait_time = 2**retry_count  # Exponential backoff
                    print(f"\nError downloading {path.name}: {e!s}")
                    print(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    sleep(wait_time)
                else:
                    print(f"\nFailed to download {path.name} after {max_retries} attempts: {e!s}")
                    raise

    # Download each model file
    checksums = verify_checksums()
    for path, is_valid in checksums.items():
        if not is_valid:
            try:
                download_with_progress(MODEL_URLS[path], Path(path), MODEL_CHECKSUMS[path])
            except Exception as e:
                print(f"Error: Failed to download {path}: {e!s}")
                sys.exit(1)


def say(text: str, config_path: str | Path = "glados_config.yaml") -> None:
    """
    Converts text to speech using the GLaDOS text-to-speech system and plays the generated audio.

    Parameters:
        text (str): The text to be spoken by the GLaDOS voice assistant.
        config_path (str | Path, optional): Path to the configuration YAML file.
            Defaults to "glados_config.yaml".

    Notes:
        - Uses a text-to-speech synthesizer to generate audio
        - Converts input text to a spoken format before synthesis
        - Plays the generated audio using the system's default sound device
        - Blocks execution until audio playback is complete

    Example:
        say("Hello, world!")  # Speaks the text using GLaDOS voice
    """
    glados_tts = tts_glados.Synthesizer()
    converter = stc.SpokenTextConverter()
    converted_text = converter.text_to_spoken(text)
    # Generate the audio to from the text
    audio = glados_tts.generate_speech_audio(converted_text)

    # Play the audio
    sd.play(audio, glados_tts.sample_rate)
    sd.wait()


def start(config_path: str | Path = "glados_config.yaml", mode: str = "audio") -> None:
    """
    Start the GLaDOS voice assistant and initialize its listening event loop.

    This function loads the GLaDOS configuration from a YAML file, creates a GLaDOS instance,
    and begins the continuous listening process for voice interactions.

    Parameters:
        config_path (str | Path, optional): Path to the configuration YAML file.
            Defaults to "glados_config.yaml" in the current directory.
        mode (str, optional): Interaction mode, either "audio" or "text". Defaults to "audio".

    Raises:
        FileNotFoundError: If the specified configuration file cannot be found.
        ValueError: If the configuration file is invalid or cannot be parsed.

    Example:
        start()  # Uses default configuration file and audio mode
        start("/path/to/custom/config.yaml", mode="text")  # Uses a custom configuration file and text mode
    """
    glados_config = GladosConfig.from_yaml(str(config_path))
    glados = Glados.from_config(glados_config)

    if mode == "audio":
        glados.start_listen_event_loop()
    elif mode == "text":
        glados.start_text()
    elif mode == "twitch":
        glados.start_auto_talk_loop()
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'audio' or 'text'.")


def tui(config_path: str | Path = "glados_config.yaml", mode: str = "audio") -> None:
    """
    Start the GLaDOS voice assistant with a terminal user interface (TUI).

    This function initializes the GLaDOS TUI application, which provides decorative
    interface elements for voice interactions.

    Parameters:
        config_path (str | Path, optional): Path to the configuration YAML file.
            Defaults to "glados_config.yaml" in the current directory.
        mode (str, optional): Interaction mode, either "audio" or "text". Defaults to "audio".
    """

    import sys

    import glados.tui as tui

    try:
        app = tui.GladosUI(mode=mode)
        app.run()
    except KeyboardInterrupt:
        sys.exit()


async def tui_twitch(config_path: str | Path = "glados_config.yaml") -> None:
    """
    Run the GladosUI in Twitch mode and listening to chat (streaming is done in the bash script directly with ffmpeg).

    Args:
        config_path (str | Path): Path to the configuration file. Defaults to "glados_config.yaml".
    """
    import os
    import glados.tui as tui
    import sys

    try:

        # Initialize the GladosUI in Twitch mode
        glados_ui = tui.GladosUI(mode="twitch")

        # Start the Twitch bot as a concurrent task
        bot_task = asyncio.create_task(start_twitch_bot(glados_ui=glados_ui))

        # Set a unique title for the current terminal window
        unique_title = f"GladosTwitchStream_{os.getpid()}"
        print(f"\033]0;{unique_title}\007", flush=True)  # Set terminal title using escape sequence

        # Give streaming time to initialize
        await asyncio.sleep(2)

        # This start the auto talk, comment it if you don't want this feature
        # Run the GladosUI application
        await glados_ui.run_async()

    except KeyboardInterrupt:
        sys.exit()

    # Wait for the bot task to complete
    await bot_task


def models_valid() -> bool:
    """
    Check the validity of all model files for the GLaDOS voice assistant.

    Verifies the integrity of model files by computing their checksums and comparing them against expected values.

    Returns:
        bool: True if all model files are valid and present, False otherwise. When False, prints a message
              instructing the user to download the required model files.
    """
    results = verify_checksums()
    if not all(results.values()):
        print("Some model files are missing or invalid. Please run 'uv run glados download'")
        return False
    return True


def main() -> None:
    """
    Command-line interface (CLI) entry point for the GLaDOS voice assistant.

    Provides several primary commands:
    - 'download': Download required model files
    - 'start': Launch the GLaDOS voice assistant
    - 'say': Generate speech from input text
    - 'tui': Start the GLaDOS voice assistant with a terminal user interface

    The function sets up argument parsing with optional configuration file paths and handles
    command execution based on user input. If no command is specified, it defaults to starting
    the assistant.

    Optional Arguments:
        --config (str): Path to configuration file, defaults to 'glados_config.yaml'
        --mode (str): Interaction mode, either 'audio' or 'text' (default: 'audio')

    Raises:
        SystemExit: If invalid arguments are provided
    """
    parser = argparse.ArgumentParser(description="GLaDOS Voice Assistant")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Download command
    subparsers.add_parser("download", help="Download model files")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start GLaDOS voice assistant")
    start_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG})",
    )
    start_parser.add_argument(
        "--mode",
        type=str,
        choices=["audio", "text", "twitch"],
        default="audio",
        help="Interaction mode: 'audio' for microphone input, 'text', 'twitch' for terminal input (default: 'audio')",
    )

    # TUI command
    tui_parser = subparsers.add_parser("tui", help="Start GLaDOS voice assistant with TUI")
    tui_parser.add_argument(
        "--mode",
        type=str,
        choices=["audio", "text", "twitch"],
        default="audio",
        help="Interaction mode: 'audio' for microphone input, 'text' for terminal input, 'twitch' for twitch input (default: 'audio')",
    )

    # Say command
    say_parser = subparsers.add_parser("say", help="Make GLaDOS speak text")
    say_parser.add_argument("text", type=str, help="Text for GLaDOS to speak")
    say_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG})",
    )

    args = parser.parse_args()

    if args.command == "download":
        download_models()
    else:
        if models_valid() is False:
            return
        if args.command == "say":
            say(args.text, args.config)
        elif args.command == "start":
            start(args.config, args.mode)
        elif args.command == "tui":
            if args.mode == "twitch":
                asyncio.run(tui_twitch())
            else:
                tui(mode=args.mode if "mode" in args else "audio")
        else:
            # Default to start if no command specified
            start(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
