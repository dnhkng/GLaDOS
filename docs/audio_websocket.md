# WebSocket Audio Backend

GLADOS can stream audio over a WebSocket instead of using your local
microphone/speaker. This lets the assistant run on a headless machine while
microphones and speakers live on the network (a browser tab, an app, or a
dedicated device).

The backend is selected through the audio factory using the same `AudioIO` ABC
as the local (sounddevice) backend — the engine code is identical either way.

## Enabling it

Select the backend in the config:

```yaml
Glados:
  audio_io: "websocket"
  audio_io_options:
    server: "127.0.0.1"     # listen address
    port: 5051              # listen port
    rooms: false            # multi-microphone room choreography (default: off)
    segregate_speakers: false
    default_room_tag: "office"
```

A ready-to-edit template is at `configs/glados_websocket_config.yaml`.

## Endpoints

The server exposes two WebSocket paths:

| Path            | Role                                              |
|-----------------|---------------------------------------------------|
| `/microphone`   | client → server: stream mic audio for VAD/ASR     |
| `/speaker`      | server → client: stream TTS audio for playback    |

Audio is raw `float32` PCM at 16 kHz. Text frames are UTF-8; audio frames are
binary.

## Speaker protocol (`/speaker`)

Server → client: `time:<unix_ts>` (start time), `sampleRate:<hz>`, then raw
binary audio.

Client → server: `played` when playback finishes, `sync_ping` (server replies
`sync_pong:<ts>`), and optionally `room:<tag>`.

On interrupt the server sends `reset` and the client must stop playback.

## Microphone protocol (`/microphone`)

Server → client: `sampleRate:<hz>` on connect.

Client → server: binary audio chunks, optionally `room:<tag>`.

The server runs VAD per client and enqueues speech with a confidence flag into
the shared sample queue consumed by ASR.

## Room choreography (opt-in via `rooms: true`)

When `rooms: true`, the backend arbitrates **multiple** clients:

- Only one microphone "owns" the input stream at a time; it yields after
  enough consecutive silent chunks (`mic_max_silence_chunks`), letting a
  different mic with voice take over.
- With `segregate_speakers: true`, audio is only sent to speakers whose room
  tag matches the active mic's room tag — so GLADOS "follows" you between rooms
  instead of playing everywhere.

With `rooms: false` (the default) each connected mic is treated as a single
source and audio is broadcast to every connected speaker — the simple
single-user case.

## Reference clients

- Browser: `tests/audio-websocket-mic.html`, `tests/audio-websocket-speaker.html`,
  and `tests/audio-websocket-both.html` (mic, speaker, and combined).
- Python: `examples/audio_websocket_client.py` is a minimal reference showing how
  to push mic audio and render received audio.
