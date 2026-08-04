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

import numpy as np
import websockets

SAMPLE_RATE = 16000


async def mic_client(host: str, port: int, timeout: float) -> None:
    """Capture from the default mic and stream float32 16 kHz audio."""
    import sounddevice as sd

    out = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        out.put_nowait(np.ascontiguousarray(indata[:, 0], dtype=np.float32).tobytes())

    async def sender():
        while True:
            await ws.send(await out.get())

    async with websockets.connect(f"ws://{host}:{port}/microphone") as ws:
        print("<-", await ws.recv())  # sampleRate:<hz>
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=512, callback=callback):
            await asyncio.wait_for(asyncio.gather(sender()), timeout)


async def speaker_client(host: str, port: int, timeout: float) -> None:
    """Render audio frames received on /speaker."""
    import sounddevice as sd

    rate = SAMPLE_RATE

    async with websockets.connect(f"ws://{host}:{port}/speaker") as ws:
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, str):
                if msg.startswith("sampleRate:"):
                    rate = int(msg.split(":", 1)[1])
                elif msg == "reset":
                    sd.stop()
            else:
                sd.play(np.frombuffer(msg, dtype=np.float32), rate)


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
