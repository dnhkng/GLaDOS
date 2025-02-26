import asyncio
import subprocess
from pathlib import Path
from typing import Optional

import twitchio
from twitchio.ext import commands

from glados.glados_ui.tui import GladosUI


class TwitchBot(commands.Bot):
    def __init__(
        self, glados_ui: GladosUI, token: str, prefix: str, initial_channels: list[str]
    ):
        super().__init__(token=token, prefix=prefix, initial_channels=initial_channels)
        self.glados_ui = glados_ui

    async def event_ready(self):
        print(f"Logged in as | {self.nick}")
        print(f"User id is | {self.user_id}")

    async def event_message(self, message: twitchio.Message):
        # Ignore messages from the bot itself
        if message.author.name.lower() == self.nick.lower():
            return

        # Send the chat message to GladosUI
        self.glados_ui.send_message_to_llm(
            user_input=message.content, user_name=message.author.name
        )

    #     # Ensure the bot processes commands as well
    #     await self.handle_commands(message)

    # @commands.command(name="hello")
    # async def hello(self, ctx: commands.Context):
    #     await ctx.send(f"Hello {ctx.author.name}!")


async def start_twitch_bot(glados_ui: GladosUI, token: str, channel: str):
    import sys

    try:
        bot = TwitchBot(
            glados_ui=glados_ui, token=token, prefix="!", initial_channels=[channel]
        )
        await bot.start()
    except KeyboardInterrupt:
        sys.exit()


def start_twitch_stream(
    ffmpeg_path: str, stream_key: str, terminal_width: int, terminal_height: int
):
    """
    Start streaming the terminal content to Twitch using ffmpeg.

    Args:
        ffmpeg_path (str): Path to the ffmpeg executable.
        stream_key (str): Twitch stream key.
        terminal_width (int): Width of the terminal to capture.
        terminal_height (int): Height of the terminal to capture.
    """
    command = [
        ffmpeg_path,
        "-f",
        "x11grab",  # Capture from X11 display
        "-video_size",
        f"{terminal_width}x{terminal_height}",  # Terminal size
        "-framerate",
        "30",  # Frame rate
        "-i",
        ":0.0",  # Display to capture (default is :0.0)
        "-f",
        "flv",  # Output format
        "-c:v",
        "libx264",  # Video codec
        "-preset",
        "veryfast",  # Encoding speed/quality tradeoff
        "-maxrate",
        "3000k",  # Max bitrate
        "-bufsize",
        "6000k",  # Buffer size
        "-pix_fmt",
        "yuv420p",  # Pixel format
        "-g",
        "60",  # Keyframe interval
        "-f",
        "flv",  # Output format
        f"rtmp://live.twitch.tv/app/{stream_key}",  # Twitch RTMP URL
    ]

    subprocess.run(command, check=True)


def run_twitch_mode(config_path: str | Path = "glados_config.yaml"):
    """
    Run the GladosUI in Twitch mode, streaming the terminal and listening to chat.

    Args:
        config_path (str | Path): Path to the configuration file. Defaults to "glados_config.yaml".
    """
    # Load Twitch credentials from config (you need to add these to your config file)
    twitch_config = GladosConfig.from_yaml(str(config_path)).get("twitch", {})
    token = twitch_config.get("token")  # Twitch OAuth token
    channel = twitch_config.get("channel")  # Twitch channel name
    stream_key = twitch_config.get("stream_key")  # Twitch stream key
    ffmpeg_path = twitch_config.get(
        "ffmpeg_path", "ffmpeg"
    )  # Path to ffmpeg executable

    if not token or not channel or not stream_key:
        raise ValueError(
            "Twitch credentials (token, channel, stream_key) must be provided in the config file."
        )

    # Start the GladosUI in Twitch mode
    glados_ui = GladosUI(mode="twitch")
    glados_ui.start_glados(mode="twitch")

    # Start the Twitch bot in a separate thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_twitch_bot(glados_ui, token, channel))

    # Start streaming the terminal content to Twitch
    terminal_width = 80  # Adjust based on your terminal size
    terminal_height = 24  # Adjust based on your terminal size
    start_twitch_stream(ffmpeg_path, stream_key, terminal_width, terminal_height)


if __name__ == "__main__":
    run_twitch_mode()
