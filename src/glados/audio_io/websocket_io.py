"""WebSocket-backed microphone capture and synchronized speaker playback."""

import asyncio
import concurrent.futures
from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Any
import uuid

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import websockets

from . import VAD
from .base import AudioIO


@dataclass
class AudioData:
    """
    Audio Data. Encapsulated here for synchronization.
    """

    data: NDArray[np.float32]
    sample_rate: int
    play_time: float
    play_time_monotonic: float
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

    def inactive(self, max_silence_chunks: int) -> bool:
        """Return whether this microphone has been silent long enough to yield."""
        return self.silence_chunks >= max_silence_chunks


class WebsocketAudioIO(AudioIO):
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
    MIC_QUEUE_MAX_CHUNKS: int = 256  # 8.2 seconds of 32 ms VAD chunks
    DEFAULT_ROOM_TAG: str = "office"  # default room tag
    ROOMS: bool = False  # enable multi-microphone room choreography / segregation
    SEGREGATE_SPEAKERS: bool = False  # default value for speaker segregation.

    def __init__(self, vad_threshold: float | None = None, options: dict[str, Any] | None = None) -> None:
        """Initialize the websocket audio I/O.

        Args:
            vad_threshold: Threshold for VAD detection (default: 0.8)
            options: backend options
              - server: Websocket listening address (default: 127.0.0.1)
              - port: Websocket listening port (default: 5051)
              - speaker_sync_delay_ms: Milliseconds added to the scheduled playback
                time for speaker synchronisation (default: 250)
              - mic_max_silence_chunks: Consecutive silent VAD chunks before the
                current microphone relinquishes control (default: 10)
              - mic_queue_max_chunks: Buffered 32 ms chunks before the oldest
                chunk is dropped (default: 256)

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
        self._mic_queue_max_chunks: int = self.MIC_QUEUE_MAX_CHUNKS
        self._default_room_tag: str = self.DEFAULT_ROOM_TAG
        self._segregate_speakers: bool = self.SEGREGATE_SPEAKERS
        self._rooms: bool = self.ROOMS

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
                    case "mic_queue_max_chunks":
                        self._mic_queue_max_chunks = int(val)
                    case "default_room_tag":
                        self._default_room_tag = str(val)
                    case "rooms":
                        if isinstance(val, bool):
                            self._rooms = val
                        else:
                            raise ValueError("rooms must be a boolean value")
                    case "segregate_speakers":
                        if isinstance(val, bool):
                            self._segregate_speakers = val
                        else:
                            raise ValueError("segregate_speakers must be a boolean value")
                    case _:
                        raise ValueError(f"Websocket backend: unsupported option '{key}'")

        if self._mic_queue_max_chunks <= 0:
            raise ValueError("mic_queue_max_chunks must be greater than zero")

        # Sample queue
        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue(
            maxsize=self._mic_queue_max_chunks
        )
        self._dropped_mic_chunks = 0

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

        # WebSocket server lifecycle, owned by the server thread.
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._server_instance: Any | None = None

        startup_future: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._server_thread = threading.Thread(
            target=self._server_thread_main,
            args=(server, port, startup_future),
            daemon=True,
        )
        self._server_thread.start()
        startup_future.result(timeout=10)

    def _server_thread_main(
        self,
        server: str,
        port: int,
        result_future: concurrent.futures.Future[None],
    ) -> None:
        """Run the asynchronous server and report startup errors to the caller."""
        try:
            asyncio.run(self._run_server(server, port, result_future))
        except Exception as ex:
            if not result_future.done():
                result_future.set_exception(ex)
            else:
                logger.debug("WebSocket audio server stopped: {}", ex)

    def start_listening(self) -> None:
        """Start capturing audio from the websocket.

        Starts capturing audio from the websocket. Each audio chunk is processed with
        the VAD model and placed in the sample queue.
        """
        self._is_listening = True

    def stop_listening(self) -> None:
        """Stop capture and release microphone ownership on the server loop."""
        self._is_listening = False

        loop = self._server_loop
        if loop is None or not loop.is_running():
            self._clear_microphone_ownership()
            return

        if threading.current_thread() is self._server_thread:
            loop.create_task(self._reset_microphone_ownership())
            return

        reset_coro = self._reset_microphone_ownership()
        try:
            reset_future = asyncio.run_coroutine_threadsafe(reset_coro, loop)
        except RuntimeError:
            reset_coro.close()
            logger.warning("WebSocket server stopped before microphone ownership could be reset")
            return

        try:
            reset_future.result(timeout=1)
        except (TimeoutError, concurrent.futures.CancelledError):
            logger.warning("Timed out while resetting WebSocket microphone ownership")

    def start_speaking(
        self,
        audio_data: NDArray[np.float32],
        sample_rate: int | None = None,
        text: str = "",
        wait: bool = False,
    ) -> None:
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
            self._playback_finished_event.wait(timeout=2.0)

        # Playback is finished
        self._playback_finished_event.clear()

        # Publish the payload and all associated state atomically. Speaker
        # tasks use _is_playing as the final "track is ready" indicator.
        with self._audio_lock:
            # allow for network jitter, time to websocket send, etc.
            sync_delay = self._speaker_sync_delay_ms / 1000
            self._audio_data = AudioData(
                np.copy(audio_data),
                sample_rate,
                time.time() + sync_delay,
                time.monotonic() + sync_delay,
                uuid.uuid4(),
            )
            self._stop_playback = False
            self._playback_was_interrupted = False
            self._is_playing = True

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

        completed = self._playback_finished_event.wait(max_timeout)

        with self._audio_lock:
            interrupted = self._playback_was_interrupted
            play_time_monotonic = (
                self._audio_data.play_time_monotonic if self._audio_data is not None else time.monotonic()
            )

        if interrupted:
            logger.debug("Playback was interrupted in Server thread")

        if not completed:
            logger.debug("Audio playback timed out, forcing interruption")
            with self._audio_lock:
                self._stop_playback = True
                self._is_playing = False
                self._playback_was_interrupted = True
                if self._audio_data is not None:
                    self._audio_data.track_id = None
                self._playback_finished_event.set()
            # Assume nothing was played because no speaker acknowledged it.
            return True, 0

        elapsed = max(0.0, time.monotonic() - play_time_monotonic)
        played_samples = elapsed * sample_rate
        percentage_played = min(int(played_samples * 100 / total_samples), 100)
        return interrupted, percentage_played

    def check_if_speaking(self) -> bool:
        """Check if audio is currently being played.

        Returns:
            bool: True if audio is currently playing, False otherwise
        """
        with self._audio_lock:
            return self._is_playing

    def stop_speaking(self) -> None:
        """Stop audio playback and clean up resources.

        Interrupts any ongoing audio playback and waits for the playback thread
        to terminate. This ensures clean resource management and prevents
        multiple overlapping playbacks.
        """
        logger.debug("Stopping speaker...")
        with self._audio_lock:
            self._stop_playback = True

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence.

        Returns:
            queue.Queue: A thread-safe queue containing tuples of
                        (audio_sample, vad_confidence)
        """
        return self._sample_queue

    def _clear_microphone_ownership(self) -> None:
        """Clear owner and silence state when no server task can be concurrent."""
        self._mic_state.current_id = None
        self._mic_state.silence_chunks = 0

    async def _reset_microphone_ownership(self) -> None:
        """Clear microphone ownership under the server-loop state lock."""
        async with self._mic_state_lock:
            self._clear_microphone_ownership()

    def _enqueue_microphone_sample(self, vad_data: NDArray[np.float32], vad_confidence: bool) -> None:
        """Enqueue recent audio while bounding memory and capture latency."""
        item = (vad_data, vad_confidence)
        try:
            self._sample_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        dropped_chunks = 0
        try:
            self._sample_queue.get_nowait()
            dropped_chunks = 1
        except queue.Empty:
            pass

        try:
            self._sample_queue.put_nowait(item)
        except queue.Full:
            # A consumer cannot refill the queue, but retain a fail-closed guard
            # if another producer is introduced later.
            dropped_chunks += 1

        self._dropped_mic_chunks += dropped_chunks
        if dropped_chunks and (self._dropped_mic_chunks == 1 or self._dropped_mic_chunks % 100 == 0):
            logger.warning(
                "Dropped {} stale microphone chunks because the consumer is behind",
                self._dropped_mic_chunks,
            )

    def close(self) -> None:
        """Stop playback and release the listening socket and server thread."""
        self.stop_listening()
        with self._audio_lock:
            self._stop_playback = True
            self._is_playing = False
            self._playback_was_interrupted = True
            if self._audio_data is not None:
                self._audio_data.track_id = None
            self._playback_finished_event.set()

        loop = self._server_loop
        server_instance = self._server_instance
        if loop is not None and loop.is_running() and server_instance is not None:
            try:
                loop.call_soon_threadsafe(server_instance.close)
            except RuntimeError:
                logger.exception("Failed to close WebSocket audio server cleanly")

        if self._server_thread.is_alive() and threading.current_thread() is not self._server_thread:
            self._server_thread.join(timeout=5)
            if self._server_thread.is_alive():
                logger.warning("WebSocket audio server thread did not stop within 5 seconds")

    def _track_is_active(self, track_id: uuid.UUID) -> bool:
        """Return whether track_id is still the currently published track."""
        with self._audio_lock:
            return self._is_playing and self._audio_data is not None and self._audio_data.track_id == track_id

    def _playback_should_stop(self) -> bool:
        """Return whether the active track has received an interruption request."""
        with self._audio_lock:
            return self._stop_playback

    async def _run_server(
        self,
        server: str,
        port: int,
        result_future: concurrent.futures.Future[None],
    ) -> None:
        """Runs the websocket server.

        Args:
            server (str): Server listen address
            port (int): Server listen port
        """
        self._mic_state_lock = asyncio.Lock()

        # re-route logging of websockets
        class LogAdapter(logging.Handler):
            """Forward standard WebSocket logs to Loguru."""

            def emit(self, record: logging.LogRecord) -> None:
                """Forward one standard logging record at its original level."""
                msg = self.format(record)
                level = record.levelname.lower()
                getattr(logger, level)(msg)

        ws_log_handler = LogAdapter()
        ws_log_handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(message)s"))

        ws_logger = logging.getLogger("websockets")
        ws_logger.addHandler(ws_log_handler)
        ws_logger.propagate = False

        self._server_loop = asyncio.get_running_loop()
        try:
            try:
                server_instance = await websockets.serve(self._server_listen, host=server, port=port)
                self._server_instance = server_instance
                if not result_future.done():
                    result_future.set_result(None)
            except Exception as ex:
                if not result_future.done():
                    result_future.set_exception(ex)
                raise

            await server_instance.wait_closed()
        finally:
            ws_logger.removeHandler(ws_log_handler)
            self._server_instance = None
            self._server_loop = None

    async def _server_listen(self, websocket: websockets.ServerConnection) -> None:
        """
        Handle incoming websocket connections.

        Args:
            websocket: Websocket connection
        """
        request = websocket.request
        if request is None:
            logger.error("WebSocket connection has no request path")
            return

        if request.path == "/speaker":
            await self._server_speaker(websocket)
        elif request.path == "/microphone":
            await self._server_microphone(websocket)
        else:
            logger.error(f"Unknown websocket path: '{request.path}'")

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
                if self._audio_data is not None and self._audio_data.track_id == track_id:
                    self._playback_was_interrupted = was_interrupted
                    self._is_playing = False
                    self._playback_finished_event.set()
                    # ensure that this is only called once
                    self._audio_data.track_id = None

        while True:
            # 1. IDLE LOOP: Check for play state, but listen for sync pings in the meantime!
            while not self.check_if_speaking():
                try:
                    # Wait for a message, but timeout quickly to check self._is_playing again
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                    await handle_default_msg(message)
                except TimeoutError:
                    continue  # Timeout expected, loop back to check `self._is_playing`
                except websockets.exceptions.ConnectionClosed:
                    return  # Client disconnected, exit the handler safely

            # check room (only meaningful when room choreography is enabled)
            if self._rooms and self._segregate_speakers:
                async with self._mic_state_lock:
                    target_room = self._mic_state.room
                if target_room != room:
                    # wait for the current playback to finish, but don't send Audio
                    while self.check_if_speaking():
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                            await handle_default_msg(message)
                        except TimeoutError:
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            return
                    continue

            # 2. AUDIO SEND PHASE
            # We acquire the lock just long enough to grab the data safely.
            with self._audio_lock:
                if self._audio_data is None:
                    continue
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
            while self._track_is_active(current_track_id) and not self._playback_should_stop():
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.05)
                    if await handle_default_msg(message) and message == "played":
                        logger.debug("Websocket: Audio played fully")
                        set_flags_once(current_track_id, False)
                        break
                except TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    set_flags_once(current_track_id, True)
                    return
            if self._playback_should_stop():
                try:
                    await websocket.send("reset")
                    logger.debug("Sent audio reset")
                except websockets.exceptions.ConnectionClosed:
                    logger.debug("Speaker disconnected before reset could be sent")
                finally:
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

        async def relinquish() -> None:
            """Release ownership only when this connection currently owns it."""
            async with self._mic_state_lock:
                if self._mic_state.current_id == client_id:
                    self._clear_microphone_ownership()

        # send sample rate
        try:
            await websocket.send("sampleRate:" + str(self.SAMPLE_RATE))
        except websockets.exceptions.ConnectionClosed:
            return

        # The default mode is deliberately single-source: the first connected
        # microphone owns input until it disconnects or listening stops.
        if not self._rooms:
            async with self._mic_state_lock:
                if self._mic_state.current_id is None:
                    self._mic_state.current_id = client_id

        while True:
            # wait for audio
            try:
                msg = await websocket.recv()
            except websockets.exceptions.ConnectionClosed:
                break

            if isinstance(msg, str) and msg.startswith("room:"):
                room = msg.split(":", maxsplit=1)[1]
            elif isinstance(msg, bytes) and self._is_listening:
                if len(msg) % np.dtype(np.float32).itemsize != 0:
                    logger.warning("Ignoring microphone frame with invalid float32 byte length: {}", len(msg))
                    continue
                # append to current_data
                data = np.frombuffer(msg, dtype=np.float32)
                current_data = np.append(current_data, data)

                # process every complete VAD window stored
                while len(current_data) >= vad_needed_samples:
                    # get data for VAD
                    vad_data = current_data[:vad_needed_samples]
                    # extra data stays for next VAD
                    current_data = current_data[vad_needed_samples:]

                    vad_value = vad_model(np.expand_dims(vad_data, 0))
                    vad_confidence = vad_value > self.vad_threshold

                    has_control = True

                    async with self._mic_state_lock:
                        if self._rooms:
                            # If no one has control, take control: because someone has to
                            if self._mic_state.current_id is None:
                                self._mic_state.current_id = client_id
                                if not vad_confidence:
                                    self._mic_state.silence_chunks = self._mic_max_silence_chunks
                            # if controlling mic is inactive and we have voice, take control
                            elif self._mic_state.inactive(self._mic_max_silence_chunks) and vad_confidence:
                                self._mic_state.current_id = client_id

                            has_control = self._mic_state.current_id == client_id

                        else:
                            # Reclaim ownership after a stop/start cycle. Only
                            # one connected microphone may feed the queue.
                            if self._mic_state.current_id is None:
                                self._mic_state.current_id = client_id
                            has_control = self._mic_state.current_id == client_id

                        # If we have control, put sample on queue
                        if has_control:
                            self._enqueue_microphone_sample(vad_data, bool(vad_confidence))
                            # always update room; a message could change it at any time
                            if self._rooms:
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
