# WebSocket Protocol

This document describes the WebSocket endpoints and communication protocol used by the audio I/O system.

## Server Configuration

- **Host**: `0.0.0.0` (configurable)
- **Port**: `5050` (configurable)
- **Audio Sample Rate**: `16000 Hz` (16 kHz)
- **Audio Format**: `float32` (NumPy dtype)

## Configuration Options

Use the `audio_io_options` key in `glados_config.yaml`.

| Option                   | Type  | Default   | Description                                                                                  |
|--------------------------|-------|-----------|----------------------------------------------------------------------------------------------|
| `server`                 | str   | `0.0.0.0` | WebSocket listen address                                                                     |
| `port`                   | int   | `5050`    | WebSocket listen port                                                                        |
| `speaker_sync_delay_ms`  | int   | `250`     | Delay added to start time for speaker sync                                                   |
| `mic_max_silence_chunks` | int   | `10`      | Silent chunks before mic relinquishes control                                                |
| `vad_threshold`          | float | `0.8`     | VAD confidence threshold (0.0 - 1.0)                                                         |
| `default_room_tag`       | str   | `office`  | Default room tag when `room:<name>` message is not sent                                      |
| `segregate_speakers`     | bool  | `False`   | If True, audio is only sent to speakers with the same room tag as the last active microphone |

## Endpoints

### `/speaker` - Audio Playback Endpoint

Used to stream audio from the server to a client for speaker playback.

#### Server → Client Messages

| Message Type | Format                  | Description                                                             |
|--------------|-------------------------|-------------------------------------------------------------------------|
| Audio Start  | `time:<unix_timestamp>` | Unix timestamp (`float`, in secs) indicating when playback should start |
| Sample Rate  | `sampleRate:<hz>`       | Audio sample rate in Hz (e.g., `sampleRate:16000`)                      |
| Audio Data   | Raw bytes               | Float32 audio samples (use `.tobytes()` to serialize)                   |

#### Client → Server Messages

| Message Type | Format        | Description                                                                            |
|--------------|---------------|----------------------------------------------------------------------------------------|
| ACK          | `played`      | Signal that audio playback is complete                                                 |
| Sync Ping    | `sync_ping`   | Request for synchronization; server responds with `sync_pong:<timestamp>`              |
| Room         | `room:<name>` | Room/location tag for the device (optional; defaults to configurable value if not set) |

#### Room Tag Segregation

If the `segregate_speakers` option is enabled (`True`), audio playback is restricted to speakers whose room tag matches the room tag of the last active microphone:

- When a microphone takes control, its room tag is recorded
- Only speakers with a matching room tag will receive audio when `segregate_speakers=True`
- Speakers with non-matching room tags will not receive audio (they may receive a `reset` message instead)
- If `segregate_speakers=False` (default), audio is broadcast to all connected speakers regardless of room tag

#### Interruption Handling

When audio playback is interrupted, the server sends:

- `reset` - Signal to reset/clean up the playback session

---

### `/microphone` - Audio Capture Endpoint

Used to stream microphone audio from a client to the server for Voice Activity Detection (VAD).

#### Server → Client Messages

| Message Type | Format            | Description                                                         |
|--------------|-------------------|---------------------------------------------------------------------|
| Sample Rate  | `sampleRate:<hz>` | Initial message; audio sample rate in Hz (e.g., `sampleRate:16000`) |

#### Client → Server Messages

| Message Type | Format        | Description                                                                            |
|--------------|---------------|----------------------------------------------------------------------------------------|
| Audio Data   | Raw bytes     | Float32 audio samples (sent with `decode=False`)                                       |
| Room         | `room:<name>` | Room/location tag for the device (optional; defaults to configurable value if not set) |

#### VAD & Mic Control

The server implements Voice Activity Detection (VAD) with the following behavior:

- **VAD Threshold**: `0.8` (configurable)
- **VAD Chunk Size**: `32 ms` (512 samples at 16 kHz)
- **Max Silence Chunks**: `10` (microphone relinquishes control after 10 silent chunks)

**Microphone Ownership Rules**:

1. Multiple clients can connect to `/microphone`
2. First client with VAD confidence > threshold takes control
3. If current mic owner becomes silent (>=10 consecutive silent chunks), other clients with voice can take control
4. On disconnect, a client relinquishes its mic control

---

## Implementation Notes

### Audio Data Serialization

**Python (Server)**:

```python
# Convert numpy array to bytes
audio_bytes = audio_data.tobytes()
```

**Python (Client)**:

```python
# Convert bytes to numpy array
audio_data = np.frombuffer(raw_bytes, dtype=np.float32)
```

### Message Flow Examples

#### Speaker Endpoint Flow

```
Client connects to /speaker
Client: room:Living Room

Server: time:1704067200.123
Server: sampleRate:16000
Server: <raw float32 audio bytes>
Client: ACK
```

#### Microphone Endpoint Flow

```
Client connects to /microphone

Client: room:Living Room
Server: sampleRate:16000

Client: <raw float32 audio bytes>
Client: <raw float32 audio bytes>
Client: <raw float32 audio bytes>
```

### Synchronization

For precise speaker synchronization, clients can use the sync ping/pong mechanism:

```
Client connects to /speaker

Client: sync_ping
Server: sync_pong:<timestamp>
```

