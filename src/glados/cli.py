import argparse
import asyncio
from hashlib import sha256
from pathlib import Path
import sys

import httpx
from rich import print as rprint
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn
import sounddevice as sd  # type: ignore

from .core.engine import Glados, GladosConfig
from .TTS import tts_glados
from .utils import spoken_text_converter as stc
from .utils.resources import resource_path

# Type aliases for clarity
type FileHash = str
type FileURL = str
type FileName = str

DEFAULT_CONFIG = resource_path("configs/glados_config.yaml")

# Details of all the models.  Each key is the file path where the model should be saved
MODEL_DETAILS: dict[FileName, dict[FileURL, FileHash]] = {
    "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/nemo-parakeet_tdt_ctc_110m.onnx",
        "checksum": "313705ff6f897696ddbe0d92b5ffadad7429a47d2ddeef370e6f59248b1e8fb5",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_encoder.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_encoder.onnx",
        "checksum": "f133a92186e63c7d4ab5b395a8e45d49f4a7a84a1d80b66f494e8205dfd57b63",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_decoder.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_decoder.onnx",
        "checksum": "415b14f965b2eb9d4b0b8517f0a1bf44a014351dd43a09c3a04d26a41c877951",
    },
    "models/ASR/parakeet-tdt-0.6b-v2_joiner.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/parakeet-tdt-0.6b-v2_joiner.onnx",
        "checksum": "846929b668a94462f21be25c7b5a2d83526e0b92a8306f21d8e336fc98177976",
    },
    "models/ASR/silero_vad_v5.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/silero_vad_v5.onnx",
        "checksum": "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396",
    },
    "models/TTS/glados.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/glados.onnx",
        "checksum": "17ea16dd18e1bac343090b8589042b4052f1e5456d42cad8842a4f110de25095",
    },
    "models/TTS/kokoro-v1.0.fp16.onnx": {
        "url": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-v1.0.fp16.onnx",
        "checksum": "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    },
    "models/TTS/kokoro-voices-v1.0.bin": {
        "url": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-voices-v1.0.bin",
        "checksum": "c5adf5cc911e03b76fa5025c1c225b141310d0c4a721d6ed6e96e73309d0fd88",
    },
    "models/TTS/phomenizer_en.onnx": {
        "url": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/phomenizer_en.onnx",
        "checksum": "b64dbbeca8b350927a0b6ca5c4642e0230173034abd0b5bb72c07680d700c5a0",
    },
}


async def download_with_progress(
    client: httpx.AsyncClient,
    url: str,
    file_path: Path,
    expected_checksum: str,
    progress: Progress,
) -> bool:
    """
    Download a single file with progress tracking and SHA-256 checksum verification.

    Returns:
        bool: True if download and verification succeeded, False otherwise
    """
    task_id = progress.add_task(f"Downloading {file_path}", status="")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    hash_sha256 = sha256()

    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            # Set total size for progress bar
            total_size = int(response.headers.get("Content-Length", 0))
            if total_size:
                progress.update(task_id, total=total_size)

            with file_path.open(mode="wb") as f:
                async for chunk in response.aiter_bytes(32768):  # 32KB chunks
                    f.write(chunk)
                    # Update the hash as we go along, for speed
                    hash_sha256.update(chunk)
                    progress.advance(task_id, len(chunk))

        # Verify checksum, and delete failed files
        actual_checksum = hash_sha256.hexdigest()
        if actual_checksum != expected_checksum:
            progress.update(task_id, status="[bold red]Checksum failed")
            Path.unlink(file_path)
            return False
        else:
            progress.update(task_id, status="[bold green]OK")
            return True

    except Exception as e:
        progress.update(task_id, status=f"[bold red]Error: {str(e)}")
        return False


async def download_models() -> int:
    """
    Main async controller for downloading all the specified models:
        - ASR model: nemo-parakeet_tdt_ctc_110m.onnx
        - VAD model: silero_vad.onnx
        - TTS model: glados.onnx
        - Phonemizer model: phomenizer_en.onnx

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    with Progress(
        TextColumn("[grey50][progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TextColumn("  {task.fields[status]}"),
    ) as progress:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Create a download task for each file
            tasks = [
                asyncio.create_task(
                    download_with_progress(client, model_info["url"], Path(path), model_info["checksum"], progress)
                )
                for path, model_info in MODEL_DETAILS.items()
            ]
            results: list[bool] = await asyncio.gather(*tasks)

    if not all(results):
        rprint("\n[bold red]Some files were not downloaded successfully")
        return 1
    rprint("\n[bold green]All files downloaded and verified successfully")
    return 0


def models_valid() -> bool:
    """
    Check the validity of all model files for the GLaDOS voice assistant.

    Verifies the integrity of model files by computing their checksums and comparing them against expected values.

    Returns:
        bool: True if all model files are valid and present, False otherwise.
    """
    for path, model_info in MODEL_DETAILS.items():
        file_path = Path(path)
        if not (file_path.exists() and sha256(file_path.read_bytes()).hexdigest() == model_info["checksum"]):
            return False
    return True


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
    glados_tts = tts_glados.SpeechSynthesizer()
    converter = stc.SpokenTextConverter()
    converted_text = converter.text_to_spoken(text)
    # Generate the audio to from the text
    audio = glados_tts.generate_speech_audio(converted_text)

    # Play the audio
    sd.play(audio, glados_tts.sample_rate)
    sd.wait()


def start(config_path: str | Path = "glados_config.yaml") -> None:
    """
    Start the GLaDOS voice assistant and initialize its listening event loop.

    This function loads the GLaDOS configuration from a YAML file, creates a GLaDOS instance,
    and begins the continuous listening process for voice interactions.

    Parameters:
        config_path (str | Path, optional): Path to the configuration YAML file.
            Defaults to "glados_config.yaml" in the current directory.

    Raises:
        FileNotFoundError: If the specified configuration file cannot be found.
        ValueError: If the configuration file is invalid or cannot be parsed.

    Example:
        start()  # Uses default configuration file
        start("/path/to/custom/config.yaml")  # Uses a custom configuration file
    """
    glados_config = GladosConfig.from_yaml(str(config_path))
    glados = Glados.from_config(glados_config)
    if glados.announcement:
        glados.play_announcement()
    glados.run()


def tui(config_path: str | Path = "glados_config.yaml") -> None:
    """
    Start the GLaDOS voice assistant with a terminal user interface (TUI).

    This function initializes the GLaDOS TUI application, which provides decorative
    interface elements for voice interactions.
    """

    import sys

    import glados.tui as tui

    try:
        app = tui.GladosUI()
        app.run()
    except KeyboardInterrupt:
        sys.exit()


def main() -> int:
    """
    Command-line interface (CLI) entry point for the GLaDOS voice assistant.

    Provides three primary commands:
    - 'download': Download required model files
    - 'start': Launch the GLaDOS voice assistant
    - 'say': Generate speech from input text

    The function sets up argument parsing with optional configuration file paths and handles
    command execution based on user input. If no command is specified, it defaults to starting
    the assistant.

    Optional Arguments:
        --config (str): Path to configuration file, defaults to 'glados_config.yaml'

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

    # TUI command
    tui_parser = subparsers.add_parser("tui", help="Start GLaDOS voice assistant with TUI")

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
        return asyncio.run(download_models())
    else:
        if not models_valid():
            print("Some model files are invalid or missing. Please run 'uv run glados download'")
            return 1
        if args.command == "say":
            say(args.text, args.config)
        elif args.command == "start":
            start(args.config)
        elif args.command == "tui":
            tui()
        else:
            # Default to start if no command specified
            start(DEFAULT_CONFIG)
        return 0


if __name__ == "__main__":
    sys.exit(main())
