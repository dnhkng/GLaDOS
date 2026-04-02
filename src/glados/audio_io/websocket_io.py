import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import websockets
from loguru import logger
import numpy as np
from numpy.typing import NDArray

from . import VAD


@dataclass
class AudioData:
    """
    Audio Data. Encapsulated here for synchronization.
    """
    data: NDArray[np.float32]
    sample_rate: int
    play_time: float
    track_id: uuid.UUID | None


@dataclass
class MicState:
    """
    Microphone State.
    Encapsulated here for synchronization.
    """
    room: str
    current_id: uuid.UUID | None = None
    silence_chunks: int = 0

    def inactive(self, max_silence_chunks: int):
        return self.silence_chunks >= max_silence_chunks


class WebsocketAudioIO:
    """Audio I/O implementation using websockets for both input and output.

    This class provides an implementation of the AudioIO interface using the
    websockets library to interact with remote clients. It handles
    real-time audio capture with voice activity detection and audio playback.
    """

    SAMPLE_RATE: int = 16000  # Sample rate for input stream
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection
    SERVER: str = "127.0.0.1"  # websockets server listen address
    PORT: int = 5051  # websockets server port
    SPEAKER_SYNC_DELAY_MS: int = 250  # Milliseconds to add to start time to account for speaker synchronisation
    MIC_MAX_SILENCE_CHUNKS: int = 10  # how many VAD chunks must be silent for a mic to relinquish control
    DEFAULT_ROOM_TAG: str = "office"  # default room tag
    SEGREGATE_SPEAKERS: bool = False  # default value for speaker segregation.

    def __init__(self, vad_threshold: float | None = None, options: dict[str, Any] | None = None) -> None:
        """Initialize the websocket audio I/O.

        Args:
            vad_threshold: Threshold for VAD detection (default: 0.8)
            options: backend options
              - server: Websocket listening address (default: 127.0.0.1)
              - port: Websocket listening port (default: 5051)
              - speaker_sync_delay_ms: Milliseconds to add to each speak start time to account for speaker synchronisation (default: 250)
              - mic_max_silence_chunks: How many consecutive VAD chunks must be silent so that the current microphone relinquishes control (default: 10)

        Raises:
            ValueError: If invalid parameters are provided
        """
        if vad_threshold is None:
            self.vad_threshold = self.VAD_THRESHOLD
        else:
            self.vad_threshold = vad_threshold

        if not 0 <= self.vad_threshold <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")

        server: str = self.SERVER
        port: int = self.PORT
        self._speaker_sync_delay_ms: int = self.SPEAKER_SYNC_DELAY_MS
        self._mic_max_silence_chunks: int = self.MIC_MAX_SILENCE_CHUNKS
        self._default_room_tag: str = self.DEFAULT_ROOM_TAG
        self._segregate_speakers: bool = self.SEGREGATE_SPEAKERS

        if options is not None:
            for key in options:
                val = options[key]
                match key:
                    case "server":
                        server = str(val)
                    case "port":
                        port = int(val)
                    case "speaker_sync_delay_ms":
                        self._speaker_sync_delay_ms = int(val)
                    case "mic_max_silence_chunks":
                        self._mic_max_silence_chunks = int(val)
                    case "default_room_tag":
                        self._default_room_tag = str(val)
                    case "segregate_speakers":
                        if isinstance(val, bool):
                            self._segregate_speakers = val
                        else:
                            raise ValueError("segregate_speakers must be a boolean value")
                    case _:
                        raise ValueError(f"Websocket backend: unsupported option '{key}'")

        # Sample queue
        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()

        # if audio is currently playing
        self._is_playing = False
        self._stop_playback = False
        # set by playback thread when playback is finished
        self._playback_finished_event = threading.Event()
        # audio payload data with lock
        self._audio_lock = threading.Lock()
        self._audio_data: AudioData | None = None
        # if the playback was interrupted by another task, this is set
        self._playback_was_interrupted: bool = False

        # if microphone is listening
        self._is_listening = False
        # microphone state: lock initialized in self._run_server
        self._mic_state_lock: asyncio.Lock
        self._mic_state = MicState(room=self._default_room_tag)

        startup_future: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._server_thread = threading.Thread(
            target=lambda s, p, f: asyncio.run(self._run_server(s, p, f)),
            args=(server, port, startup_future),
            daemon=True
        )
        self._server_thread.start()
        startup_future.result(timeout=10)

    def start_listening(self) -> None:
        """Start capturing audio from the websocket.

        Starts capturing audio from the websocket. Each audio chunk is processed with
        the VAD model and placed in the sample queue.
        """
        self._is_listening = True

    def stop_listening(self) -> None:
        """Stop capturing audio"""
        self._is_listening = False

    def start_speaking(self, audio_data: NDArray[np.float32], sample_rate: int | None = None, text: str = "", wait: bool = False) -> None:
        """Play audio through the system speakers.

        Parameters:
            audio_data: The audio data to play as a numpy float32 array
            sample_rate: The sample rate of the audio data in Hz
            text: Optional text associated with the audio (not used by this implementation)
            wait: Optionally wait for the audio_data to be spoken
        """
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0 or audio_data.dtype != np.float32:
            raise ValueError("Invalid audio data")

        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        if self._is_playing:
            # Stop any existing playback and wait for finish
            self.stop_speaking()
            self._playback_finished_event.wait()

        # Playback is finished
        self._playback_finished_event.clear()

        # Lock, set data, unlock
        with self._audio_lock:
            # allow for network jitter, time to websocket send, etc.
            play_time = time.time() + (self._speaker_sync_delay_ms / 1000)
            self._audio_data = AudioData(audio_data, sample_rate, play_time, uuid.uuid4())

        # set state
        self._stop_playback = False
        self._is_playing = True
        self._playback_was_interrupted = False

        logger.debug("Scheduled audio playback")

        if wait:
            max_timeout = (len(audio_data) / sample_rate) + (self._speaker_sync_delay_ms / 1000.0) + 1.0
            self._playback_finished_event.wait(timeout=max_timeout)

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """
        Monitor audio playback progress and return completion status with interrupt detection.

        Streams audio samples and actively tracks the number of samples
        that have been played. The playback can be interrupted.

        Args:
            total_samples (int): Total number of samples in the audio data being played.
            sample_rate (int): Sample rate of the audio data in Hz.

        Returns:
            tuple[bool, int]: A tuple containing:
                - bool: True if playback was interrupted, False if completed normally
                - int: Percentage of audio played (0-100)
        """
        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        # wait for finish
        max_timeout = (total_samples / sample_rate) + (self._speaker_sync_delay_ms / 1000.0) + 1.0

        now = time.monotonic()
        completed = self._playback_finished_event.wait(max_timeout)
        interrupted = self._playback_was_interrupted
        elapsed = time.monotonic() - now

        if interrupted:
            logger.debug("Playback was interrupted in Server thread")

        if not completed:
            logger.debug("Audio playback timed out, forcing interruption")
            # Assume nothing was played because no speaker was there
            return True, 0

        played_samples = elapsed * sample_rate
        percentage_played = min(int(played_samples * 100 / total_samples), 100)
        return interrupted, percentage_played

    def check_if_speaking(self) -> bool:
        """Check if audio is currently being played.

        Returns:
            bool: True if audio is currently playing, False otherwise
        """
        return self._is_playing

    def stop_speaking(self) -> None:
        """Stop audio playback and clean up resources.

        Interrupts any ongoing audio playback and waits for the playback thread
        to terminate. This ensures clean resource management and prevents
        multiple overlapping playbacks.
        """
        logger.debug("Stopping speaker...")
        self._stop_playback = True

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence.

        Returns:
            queue.Queue: A thread-safe queue containing tuples of
                        (audio_sample, vad_confidence)
        """
        return self._sample_queue

    async def _run_server(self, server: str, port: int, result_future: concurrent.futures.Future) -> None:
        """Runs the websocket server.

        Args:
            server (str): Server listen address
            port (int): Server listen port
        """
        self._mic_state_lock = asyncio.Lock()

        # re-route logging of websockets
        class LogAdapter(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                msg = self.format(record)
                level = record.levelname.lower()
                getattr(logger, level)(msg)

        ws_log_handler = LogAdapter()
        ws_log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(message)s"))

        ws_logger = logging.getLogger("websockets")
        ws_logger.addHandler(ws_log_handler)
        ws_logger.propagate = False

        try:
            server = await websockets.serve(self._server_listen, host=server, port=port)
            result_future.set_result(None)
        except OSError as ex:
            result_future.set_exception(ex)
            raise

        await server.serve_forever()

    async def _server_listen(self, websocket: websockets.ServerConnection) -> None:
        """
        Handle incoming websocket connections.

        Args:
            websocket: Websocket connection
        """
        if websocket.request.path == "/speaker":
            await self._server_speaker(websocket)
        elif websocket.request.path == "/microphone":
            await self._server_microphone(websocket)
        else:
            logger.error(f"Unknown websocket path: '{websocket.request.path}'")

    async def _server_speaker(self, websocket: websockets.ServerConnection) -> None:
        """
        Handle incoming websocket connections for speaker output.

        Args:
            websocket: Websocket connection
        """

        room = self._default_room_tag

        async def handle_default_msg(ws_msg: str | bytes) -> bool:
            """Handle the default ws messages. Returns True if the message is not a default message"""
            if ws_msg == "sync_ping":
                await websocket.send(f"sync_pong:{time.time()}")
                return False
            elif isinstance(ws_msg, str) and ws_msg.startswith("room:"):
                nonlocal room
                room = ws_msg.split(":", maxsplit=1)[1]
                return False
            return True

        def set_flags_once(track_id: uuid.UUID, was_interrupted: bool) -> None:
            """
            Set flags that audio was played if the given track_id matches the currently stored track_id.
            If flags are set, the track_id is cleared from self._audio_data.
            This ensures that the flags are only set by 1 speaker task.

            Args:
                track_id: ID of the audio track
                was_interrupted: If the audio was interrupted (as interpreted by this task).
            """
            assert track_id is not None

            with self._audio_lock:
                if self._audio_data.track_id == track_id:
                    self._playback_was_interrupted = was_interrupted
                    self._is_playing = False
                    self._playback_finished_event.set()
                    # ensure that this is only called once
                    self._audio_data.track_id = None

        while True:
            # 1. IDLE LOOP: Check for play state, but listen for sync pings in the meantime!
            while not self._is_playing:
                try:
                    # Wait for a message, but timeout quickly to check self._is_playing again
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                    await handle_default_msg(message)
                except asyncio.TimeoutError:
                    continue  # Timeout expected, loop back to check `self._is_playing`
                except websockets.exceptions.ConnectionClosed:
                    return  # Client disconnected, exit the handler safely

            # check room
            if self._segregate_speakers:
                async with self._mic_state_lock:
                    target_room = self._mic_state.room
                if target_room != room:
                    # wait for the current playback to finish, but don't send Audio
                    while self._is_playing:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                            await handle_default_msg(message)
                        except asyncio.TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            return
                    continue

            # 2. AUDIO SEND PHASE
            # We acquire the lock just long enough to grab the data safely.
            with self._audio_lock:
                play_time = self._audio_data.play_time
                sample_rate = self._audio_data.sample_rate
                audio_data_bytes = self._audio_data.data.tobytes()
                sample_count = len(self._audio_data.data)
                current_track_id = self._audio_data.track_id

            # Audio with no track ID should not be played
            if current_track_id is None:
                continue

            try:
                # Send timestamp, then sample rate, then bytes
                await websocket.send("time:" + str(play_time))
                await websocket.send("sampleRate:" + str(sample_rate))
                await websocket.send(audio_data_bytes)

                logger.debug(f"Playing audio with sample rate: {sample_rate} Hz, length: {sample_count} samples")
            except websockets.exceptions.ConnectionClosed:
                set_flags_once(current_track_id, True)
                return

            # 3. WAITING PHASE
            while not self._stop_playback:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                    if await handle_default_msg(message) and message == "played":
                        logger.debug("Websocket: Audio played fully")
                        set_flags_once(current_track_id, False)
                        break
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    set_flags_once(current_track_id, True)
                    return
            else:
                # self._stop_playback is true
                await websocket.send("reset")
                logger.debug("Sent audio reset")
                set_flags_once(current_track_id, True)

    async def _server_microphone(self, websocket: websockets.ServerConnection) -> None:
        """
        Handle incoming websocket connections for microphone input.

        Args:
            websocket: Websocket connection
        """
        # unique ID for the client
        client_id = uuid.uuid4()
        # VAD is per microphone because it stores context
        vad_model = VAD()
        # needed amount of samples for VAD
        vad_needed_samples = self.SAMPLE_RATE * self.VAD_SIZE // 1000
        # currently stored samples
        current_data = np.empty((0,), dtype=np.float32)
        # room of the mic
        room = self._default_room_tag

        async def relinquish():
            async with self._mic_state_lock:
                if self._mic_state.current_id == client_id:
                    self._mic_state.current_id = None

        # send sample rate
        try:
            await websocket.send("sampleRate:" + str(self.SAMPLE_RATE))
        except websockets.exceptions.ConnectionClosed:
            return

        while True:
            # wait for audio
            try:
                msg = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                break

            if isinstance(msg, str) and msg.startswith("room:"):
                room = msg.split(":", maxsplit=1)[1]
            elif isinstance(msg, bytes) and self._is_listening:
                # append to current_data
                data = np.frombuffer(msg, dtype=np.float32)
                current_data = np.append(current_data, data)

                # if enough current data is there, run it through the VAD
                if len(current_data) >= vad_needed_samples:
                    # get data for VAD
                    vad_data = current_data[:vad_needed_samples]
                    # extra data stays for next VAD
                    current_data = current_data[vad_needed_samples:]

                    vad_value = vad_model(np.expand_dims(vad_data, 0))
                    vad_confidence = vad_value > self.vad_threshold

                    async with self._mic_state_lock:
                        # If no one has control, take control: because someone has to
                        if self._mic_state.current_id is None:
                            self._mic_state.current_id = client_id
                            if not vad_confidence:
                                self._mic_state.silence_chunks = self._mic_max_silence_chunks
                        # if controlling mic is inactive and we have voice, take control
                        elif self._mic_state.inactive(self._mic_max_silence_chunks) and vad_confidence:
                            self._mic_state.current_id = client_id

                        # If we have control, put sample on queue
                        if self._mic_state.current_id == client_id:
                            self._sample_queue.put((vad_data, bool(vad_confidence)))
                            # always update room; a message could change it at any time
                            self._mic_state.room = room

                            if vad_confidence:
                                # also acts as init
                                self._mic_state.silence_chunks = 0
                            else:
                                self._mic_state.silence_chunks += 1

            if not self._is_listening:
                # reset when not listening
                current_data = np.empty((0,), dtype=np.float32)
                vad_model.reset_states()
                await relinquish()

        # relinquish control on connection exit
        await relinquish()
