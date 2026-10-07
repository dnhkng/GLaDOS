"""Controlled and live Sonora AEC3 probe; production GLaDOS must be stopped for live mode.

Build: cargo +1.91.0 build --release --manifest-path examples/sonora_aec_probe/Cargo.toml
Run: .venv/bin/python examples/benchmark_sonora_aec.py --render FAR.wav --work-dir /tmp/aec
Live additionally needs --live --input-device N --output-device N and cue WAVs in work-dir:
instructions.wav, talk.wav, near.wav, stop.wav. Recordings stay in work-dir.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import queue
import subprocess
import threading
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort
from scipy import signal
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "examples/sonora_aec_probe/target/release/sonora-aec-probe"
REV = "413f64e7404529aaae6980613527f0d6acb21ce7"
Audio = NDArray[np.float32]


class ClockTimes(Protocol):
    outputBufferDacTime: float  # noqa: N815 -- PortAudio's field names.
    inputBufferAdcTime: float  # noqa: N815 -- PortAudio's field names.


def load(path: Path, rate: int) -> Audio:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    divisor = math.gcd(rate, source_rate)
    return signal.resample_poly(audio, rate // divisor, source_rate // divisor).astype(np.float32)


def normalized(audio: Audio, peak: float) -> Audio:
    return audio * (peak / max(float(np.max(np.abs(audio))), 1e-8))


def power(audio: Audio) -> float:
    return float(np.mean(np.square(audio.astype(np.float64))))


def reduction(before: Audio, after: Audio) -> float:
    return 10 * math.log10(max(power(before), 1e-20) / max(power(after), 1e-20))


def speech_quality(reference: Audio, audio: Audio, rate: int) -> dict[str, float]:
    """Align by <=50 ms. Projection gain also includes any filter phase mismatch."""
    correlation = signal.correlate(audio, reference, mode="full", method="fft")
    center = len(reference) - 1
    bound = rate // 20
    lag = int(np.argmax(correlation[center - bound : center + bound + 1])) - bound
    if lag > 0:
        audio, reference = audio[lag:], reference[:-lag]
    elif lag < 0:
        audio, reference = audio[:lag], reference[-lag:]
    gain = float(
        np.dot(audio.astype(np.float64), reference) / max(np.dot(reference.astype(np.float64), reference), 1e-20)
    )
    target = gain * reference
    return {
        "si_sdr_db": reduction(target, audio - target),
        "projection_gain_db": 20 * math.log10(max(abs(gain), 1e-10)),
        "power_change_db": -reduction(reference, audio),
        "lag_ms": lag * 1000 / rate,
    }


class Silero:
    """CPU-only runner with the same 512/64 sample framing and state as production VAD."""

    def __init__(self) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(ROOT / "models/ASR/silero_vad_16k_op15.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )

    def run(self, audio: Audio, rate: int) -> NDArray[np.float64]:
        divisor = math.gcd(16000, rate)
        audio = signal.resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        audio = np.pad(audio, (0, (-len(audio)) % 512))
        state = np.zeros((2, 1, 128), np.float32)
        context = np.zeros((1, 64), np.float32)
        result = []
        for start in range(0, len(audio), 512):
            batch = np.concatenate([context, audio[None, start : start + 512]], axis=1)
            value, state = self.session.run(None, {"input": batch, "state": state, "sr": np.array(16000, np.int64)})
            result.append(float(value.reshape(-1)[0]))
            context = batch[:, -64:]
        return np.array(result)


def vad_summary(probabilities: NDArray[np.float64], start: float, end: float) -> dict[str, object]:
    mask = probabilities[int(start / 0.032) : int(end / 0.032)] > 0.8
    longest = run = 0
    for voiced in mask:
        run = run + 1 if voiced else 0
        longest = max(longest, run)
    return {
        "voiced_fraction": float(np.mean(mask)),
        "longest_voiced_run_frames": longest,
        "reaches_playback_interruption_gate": longest >= 5,
    }


def process(
    work: Path, name: str, render: Audio, capture: Audio, rate: int, delay: int = 0, bypass: bool = False
) -> tuple[Audio, dict[str, object]]:
    paths = [work / f"{name}-{kind}.wav" for kind in ["render", "capture", "clean"]]
    for path, audio in zip(paths[:2], [render, capture], strict=True):
        sf.write(path, audio, rate, subtype="FLOAT")
    command = [str(BINARY), "file", *map(str, paths), str(delay)]
    if bypass:
        command.append("bypass")
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    timing = json.loads(result.stderr.strip().splitlines()[-1])
    clean, _ = sf.read(paths[2], dtype="float32")
    return clean, timing


def offline(args: argparse.Namespace, vad: Silero) -> list[dict[str, object]]:
    cases = []
    for rate in [16000, 48000]:
        far = normalized(load(args.render, rate), 0.30)
        near_source = normalized(load(ROOT / "data/0.wav", rate), 0.20)
        count = 40 * rate
        render = np.resize(far, count).astype(np.float32)
        near = np.zeros(count, np.float32)
        near[15 * rate : 30 * rate] = np.resize(near_source, 15 * rate)
        # Match the AEC high-pass and filter-bank response, which otherwise biases SI-SDR.
        near_reference, _ = process(args.work_dir, f"near-reference-{rate}", np.zeros_like(near), near, rate)
        for delay_ms in [0, 40, 80, 200]:
            echo = np.zeros(count, np.float32)
            for extra_ms, gain in [(0, 0.60), (30, 0.20), (80, 0.10)]:
                offset = int((delay_ms + extra_ms) * rate / 1000)
                echo[offset:] += gain * render[: count - offset]
            capture = echo + near
            clean, timing = process(args.work_dir, f"offline-{rate}-{delay_ms}", render, capture, rate)
            section = slice(15 * rate, 30 * rate)
            raw_vad, clean_vad, near_vad = [vad.run(audio, rate) for audio in [capture, clean, near]]
            truth = near_vad[int(15 / 0.032) : int(30 / 0.032)] > 0.8
            detected = clean_vad[int(15 / 0.032) : int(30 / 0.032)] > 0.8
            row = {
                "rate": rate,
                "echo_delay_ms": delay_ms,
                "automatic_delay_estimation": True,
                "echo_only_reduction_db": reduction(capture[10 * rate : 15 * rate], clean[10 * rate : 15 * rate]),
                "recovery_reduction_db": reduction(capture[35 * rate :], clean[35 * rate :]),
                "double_talk_raw": speech_quality(near[section], capture[section], rate),
                "double_talk_clean": speech_quality(near[section], clean[section], rate),
                "double_talk_clean_vs_processed_near_reference": speech_quality(
                    near_reference[section], clean[section], rate
                ),
                "near_to_echo_db": reduction(near[section], echo[section]),
                "near_vad_recall": float(np.mean(detected[truth])) if truth.any() else None,
                "echo_only_raw_vad": vad_summary(raw_vad, 10, 15),
                "echo_only_clean_vad": vad_summary(clean_vad, 10, 15),
                "timing": timing,
            }
            cases.append(row)
            print(json.dumps(row), flush=True)
        clean, timing = process(
            args.work_dir, f"near-only-{rate}", np.zeros(count, np.float32), np.resize(near_source, count), rate
        )
        cases.append(
            {
                "case": "near_only",
                "rate": rate,
                **speech_quality(np.resize(near_source, count)[10 * rate :], clean[10 * rate :], rate),
                "timing": timing,
            }
        )
        silence = np.zeros(rate, np.float32)
        clean, _ = process(args.work_dir, f"silence-{rate}", silence, silence, rate)
        if np.max(np.abs(clean)) != 0:
            raise RuntimeError("silence produced nonzero output")
        bypass, _ = process(args.work_dir, f"bypass-{rate}", render, capture, rate, bypass=True)
        if not np.array_equal(bypass, capture):
            raise RuntimeError("bypass is not sample-identical")
    return cases


def level_sweep(args: argparse.Namespace, vad: Silero) -> list[dict[str, object]]:
    """48 kHz/80 ms echo; distinguish weak speech from louder nearby speakers."""
    rate, count = 48000, 40 * 48000
    render = np.resize(normalized(load(args.render, rate), 0.30), count).astype(np.float32)
    echo = np.zeros_like(render)
    for delay, gain in [(80, 0.60), (110, 0.20), (160, 0.10)]:
        offset = int(delay * rate / 1000)
        echo[offset:] += gain * render[: count - offset]
    source = load(ROOT / "data/0.wav", rate)
    section = slice(15 * rate, 30 * rate)
    near = np.zeros_like(render)
    near[section] = np.resize(source, 15 * rate)
    result = []
    for snr in [-10, 0, 6]:
        scaled_near = near * math.sqrt(power(echo[section]) / power(near[section])) * 10 ** (snr / 20)
        # Uniform scaling prevents clipping while preserving the requested ratio.
        scale = min(1.0, 0.85 / float(np.max(np.abs(echo + scaled_near))))
        playback, capture, target = render * scale, (echo + scaled_near) * scale, scaled_near * scale
        clean, timing = process(args.work_dir, f"level-{snr}", playback, capture, rate)
        reference, _ = process(args.work_dir, f"level-reference-{snr}", np.zeros_like(target), target, rate)
        truth = vad.run(reference, rate)[int(15 / 0.032) : int(30 / 0.032)] > 0.8
        detection = vad.run(clean, rate)[int(15 / 0.032) : int(30 / 0.032)] > 0.8
        result.append(
            {
                "near_to_echo_db": snr,
                "peak_capture": float(np.max(np.abs(capture))),
                "quality_vs_processed_near_reference": speech_quality(reference[section], clean[section], rate),
                "vad_recall_vs_processed_near_reference": float(np.mean(detection[truth])) if truth.any() else None,
                "timing": timing,
            }
        )
    print("LEVEL SWEEP", json.dumps(result), flush=True)
    return result


def synthesize(work: Path) -> Path:
    """Use the project's existing TTS API; no production code or provider monkeypatches."""
    from glados.TTS.tts_glados import SpeechSynthesizer

    synth = SpeechSynthesizer()
    texts = {
        "far": (
            "This is an acoustic echo cancellation test. I will keep speaking so that the microphone can hear "
            "the loudspeaker. The system should remove my voice from the microphone recording. It should still "
            "hear a person who speaks at the same time. Echo cancellation is an essential part of a voice assistant. "
            "The cake is a lie, but these measurements should be real."
        ),
        "instructions": (
            "Please remain quiet at first. When you hear begin speaking now, talk continuously over my voice. "
            "At continue speaking, keep talking during the silence. Finish only when I say you can be quiet now."
        ),
        "talk": "Begin speaking now. Please talk continuously over my voice.",
        "near": "Continue speaking. Keep talking during the silence.",
        "stop": "You can be quiet now. The recording is finished.",
    }
    for name, text in texts.items():
        audio = synth.generate_speech_audio(text).reshape(-1)
        sf.write(work / f"{name}.wav", audio, synth.sample_rate, subtype="FLOAT")
    return work / "far.wav"


def live(args: argparse.Namespace, vad: Silero) -> dict[str, object]:
    """Full duplex capture; native AEC runs in a worker, outside the PortAudio callback.

    Python copies/queues are suitable for this probe, not the planned Rust audio engine.
    Cleaned microphone audio is recorded, never routed back to the loudspeaker.
    """
    import sounddevice as sd

    rate, frame = 48000, 480
    far = normalized(load(args.render, rate), 0.25)
    cues = {
        key: normalized(load(args.work_dir / f"{key}.wav", rate), 0.25)
        for key in ["instructions", "talk", "near", "stop"]
    }
    # Echo convergence, double talk, then isolated near speech. Spoken cues delimit phases.
    render = np.zeros(44 * rate, np.float32)
    render[2 * rate : 17 * rate] = np.resize(far, 15 * rate)
    render[17 * rate : 17 * rate + len(cues["talk"])] = cues["talk"]
    overlap_start = 17 * rate + len(cues["talk"])
    render[overlap_start : 32 * rate] = np.resize(far, 32 * rate - overlap_start)
    render[32 * rate : 32 * rate + len(cues["near"])] = cues["near"]
    render[40 * rate : 40 * rate + len(cues["stop"])] = cues["stop"][: 4 * rate]
    print("Playing instructions; live test starts next. Keep quiet until 'begin speaking now'.", flush=True)
    sd.play(cues["instructions"], rate, device=args.output_device, blocking=True)
    sd.sleep(1000)
    items = queue.Queue(maxsize=100)
    records, faults, clock_delays = [], [], []
    native = subprocess.Popen(
        [str(BINARY), "stream", str(rate), "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    worker_errors = []

    def worker() -> None:
        try:
            while True:
                item = items.get()
                if item is None:
                    break
                playback, capture = item
                data = np.concatenate([playback, capture]).astype("<f4").tobytes()
                view = memoryview(data)
                while view:
                    written = native.stdin.write(view)
                    if not written:
                        raise RuntimeError("native worker closed stdin")
                    view = view[written:]
                result = bytearray()
                while len(result) < frame * 4:
                    part = native.stdout.read(frame * 4 - len(result))
                    if not part:
                        raise RuntimeError("native worker closed stdout")
                    result.extend(part)
                records.append((playback, capture, np.frombuffer(result, dtype="<f4").copy()))
        except Exception as exc:
            worker_errors.append(str(exc))
        finally:
            native.stdin.close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    cursor = 0

    def callback(indata: Audio, outdata: Audio, frames: int, timestamps: ClockTimes, status: object) -> None:
        nonlocal cursor
        if status:
            faults.append(str(status))
        if frames != frame:
            faults.append(f"unexpected callback size {frames}")
            raise sd.CallbackAbort from None
        if cursor >= len(render):
            outdata.fill(0)
            raise sd.CallbackStop
        playback = render[cursor : cursor + frame]
        outdata[:, 0] = playback
        clock_delays.append((timestamps.outputBufferDacTime - timestamps.inputBufferAdcTime) * 1000)
        try:
            items.put_nowait((playback.copy(), indata[:, 0].copy()))
        except queue.Full:
            faults.append("AEC worker queue overflow")
            raise sd.CallbackAbort from None
        cursor += frame

    try:
        with sd.Stream(
            device=(args.input_device, args.output_device),
            samplerate=rate,
            blocksize=frame,
            channels=(1, 1),
            dtype="float32",
            latency=0.02,
            callback=callback,
        ) as stream:
            latency = list(stream.latency)
            print(
                "LIVE RECORDING STARTED",
                sd.query_devices(args.input_device)["name"],
                "->",
                sd.query_devices(args.output_device)["name"],
                "latency",
                latency,
                flush=True,
            )
            while stream.active and not worker_errors:
                sd.sleep(100)
    finally:
        items.put(None, timeout=5)
        thread.join(timeout=10)
        if thread.is_alive():
            native.kill()
            raise RuntimeError("native worker hung")
        native.wait(timeout=5)
    if worker_errors or native.returncode:
        raise RuntimeError(f"live worker failed: {worker_errors}; {native.stderr.read().decode()}")
    timing = json.loads(native.stderr.read().decode().strip().splitlines()[-1])
    playback, capture, clean = [np.concatenate([row[i] for row in records]) for i in range(3)]
    for name, audio in zip(["render", "capture", "clean"], [playback, capture, clean], strict=True):
        sf.write(args.work_dir / f"live-{name}.wav", audio, rate, subtype="FLOAT")
    raw_vad, clean_vad = [vad.run(audio, rate) for audio in [capture, clean]]
    result = {
        "input_device": dict(sd.query_devices(args.input_device)),
        "output_device": dict(sd.query_devices(args.output_device)),
        "latency_seconds": latency,
        "callback_delay_median_ms": float(np.median(clock_delays)),
        "callback_faults": faults,
        "seconds_recorded": len(capture) / rate,
        "input_peak": float(np.max(np.abs(capture))),
        "echo_only_reduction_db": reduction(capture[10 * rate : 16 * rate], clean[10 * rate : 16 * rate]),
        "echo_only_raw_vad": vad_summary(raw_vad, 10, 16),
        "echo_only_clean_vad": vad_summary(clean_vad, 10, 16),
        "double_talk_raw_vad": vad_summary(raw_vad, 24, 31),
        "double_talk_clean_vad": vad_summary(clean_vad, 24, 31),
        "near_only_raw_vad": vad_summary(raw_vad, 36, 39),
        "near_only_clean_vad": vad_summary(clean_vad, 36, 39),
        "timing": timing,
        "limitation": "Live double-talk has no clean reference; listen to recording to assess intelligibility.",
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", type=Path)
    parser.add_argument(
        "--synthesize", action="store_true", help="generate playback and spoken cues using local GLaDOS TTS"
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--input-device", type=int)
    parser.add_argument("--output-device", type=int)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.live_only and not args.live:
        parser.error("--live-only requires --live")
    if args.synthesize:
        args.render = synthesize(args.work_dir)
    if args.render is None:
        parser.error("provide --render or --synthesize")
    if not BINARY.exists():
        parser.error("build the Rust probe first")
    vad = Silero()
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "sonora_revision": REV,
        "platform": platform.platform(),
        "render_sha256": hashlib.sha256(args.render.read_bytes()).hexdigest(),
        "synthetic_setup": (
            "40s, GLaDOS playback repeated; data/0.wav near speech at 15-30s; "
            "linear echo taps .60/.20/.10 at delay/+30/+80ms; no added noise or clock drift; auto delay"
        ),
        "work_dir": str(args.work_dir),
        "vad_threshold": 0.8,
        "metric_note": (
            "Raw-reference SI-SDR includes high-pass/filter-bank phase effects; processed-near-reference "
            "metrics isolate overlap suppression better. A VAD gate is a screening metric, "
            "not the full production listener."
        ),
        "offline": [] if args.live_only else offline(args, vad),
        "level_sweep": [] if args.live_only else level_sweep(args, vad),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    if args.live:
        result["live"] = live(args, vad)
        args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
