"""Minimal WebSocket audio client for the GLADOS websocket backend.

Shows the client side of the protocol in ``docs/audio_websocket.md``: stream a
microphone to ``/microphone`` and/or render audio received on ``/speaker``.

Usage:
    python examples/audio_websocket_client.py mic     --host 127.0.0.1 --port 5051
    python examples/audio_websocket_client.py speaker --host 127.0.0.1 --port 5051

Requires: websockets, numpy, sounddevice
"""

import argparse
import asyncio
import time

import numpy as np
import websockets

SAMPLE_RATE = 16000


async def mic_client(host: str, port: int, timeout: float) -> None:
    """Capture from the default mic and stream float32 16 kHz audio."""
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    out: asyncio.Queue[bytes] = asyncio.Queue(maxsize=8)

    def enqueue(data: bytes) -> None:
        if out.full():
            # Keep latency bounded: discard the oldest unsent microphone block.
            out.get_nowait()
            out.task_done()
        out.put_nowait(data)

    def callback(
        indata: np.ndarray,
        _frames: int,
        _time_info: object,
        _status: object,
    ) -> None:
        data = np.ascontiguousarray(indata[:, 0], dtype=np.float32).tobytes()
        loop.call_soon_threadsafe(enqueue, data)

    async def sender() -> None:
        while True:
            data = await out.get()
            try:
                await ws.send(data)
            finally:
                out.task_done()

    async with websockets.connect(f"ws://{host}:{port}/microphone") as ws:
        print("<-", await ws.recv())  # sampleRate:<hz>
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=512, callback=callback):
            await asyncio.wait_for(sender(), timeout)


async def speaker_client(host: str, port: int, timeout: float) -> None:
    """Render audio frames received on /speaker."""
    import sounddevice as sd

    rate = SAMPLE_RATE
    play_time = time.time()
    playback_task: asyncio.Task[None] | None = None

    async def stop_playback() -> None:
        nonlocal playback_task
        if playback_task is None:
            return
        sd.stop()
        playback_task.cancel()
        await asyncio.gather(playback_task, return_exceptions=True)
        playback_task = None

    async with websockets.connect(f"ws://{host}:{port}/speaker") as ws:

        async def play(data: bytes, scheduled_time: float, sample_rate: int) -> None:
            await asyncio.sleep(max(0.0, scheduled_time - time.time()))
            sd.play(np.frombuffer(data, dtype=np.float32), sample_rate)
            await asyncio.to_thread(sd.wait)
            await ws.send("played")

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout)
                except TimeoutError:
                    break
                if isinstance(msg, str):
                    if msg.startswith("time:"):
                        play_time = float(msg.split(":", 1)[1])
                    elif msg.startswith("sampleRate:"):
                        rate = int(msg.split(":", 1)[1])
                    elif msg == "reset":
                        await stop_playback()
                else:
                    await stop_playback()
                    playback_task = asyncio.create_task(play(msg, play_time, rate))
        finally:
            await stop_playback()


def main() -> None:
    ap = argparse.ArgumentParser(description="GLADOS websocket audio client")
    sub = ap.add_subparsers(dest="mode", required=True)

    mic_p = sub.add_parser("mic", help="stream mic audio to /microphone")
    mic_p.add_argument("--host", default="127.0.0.1")
    mic_p.add_argument("--port", type=int, default=5051)
    mic_p.add_argument("--timeout", type=int, default=60)

    spk_p = sub.add_parser("speaker", help="render audio from /speaker")
    spk_p.add_argument("--host", default="127.0.0.1")
    spk_p.add_argument("--port", type=int, default=5051)
    spk_p.add_argument("--timeout", type=int, default=60)

    args = ap.parse_args()

    if args.mode == "mic":
        asyncio.run(mic_client(args.host, args.port, args.timeout))
    else:
        asyncio.run(speaker_client(args.host, args.port, args.timeout))


if __name__ == "__main__":
    main()
