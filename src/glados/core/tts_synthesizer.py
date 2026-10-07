from collections.abc import Callable
import queue
import threading
import time

from loguru import logger
import numpy as np

from ..observability import ObservabilityBus, trim_message
from ..TTS import SpeechSynthesizerProtocol
from ..utils import spoken_text_converter as stc
from .audio_data import AudioMessage
from .speech_markup import SpeechMarkupParser, SpeechText


class TextToSpeechSynthesizer:
    """
    A thread that synthesizes text to speech using a TTS model and a spoken text converter.
    It reads text from a queue, processes it, generates audio, and puts the audio messages into an output queue.
    This class is designed to run in a separate thread, continuously checking for new text to
    synthesize until a shutdown event is set.
    """

    def __init__(
        self,
        tts_input_queue: queue.Queue[str | SpeechText],
        audio_output_queue: queue.Queue[AudioMessage],
        tts_model: SpeechSynthesizerProtocol,
        stc_instance: stc.SpokenTextConverter,
        shutdown_event: threading.Event,
        pause_time: float,
        tts_muted_event: threading.Event | None = None,
        observability_bus: ObservabilityBus | None = None,
        quiet_mode: Callable[[], bool] = lambda: False,
        quiet_generation: Callable[[], int] = lambda: 0,
        on_response_ready: Callable[[int, str], None] = lambda generation, reason: None,
        autonomy_generation: Callable[[], int] = lambda: 0,
        autonomy_enabled: Callable[[], bool] = lambda: True,
    ) -> None:
        self.tts_input_queue = tts_input_queue
        self.audio_output_queue = audio_output_queue
        self.tts_model = tts_model
        self.stc = stc_instance
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time
        self._tts_muted_event = tts_muted_event
        self._observability_bus = observability_bus
        self._on_response_ready = on_response_ready
        self._quiet_mode, self._quiet_generation = quiet_mode, quiet_generation
        self._autonomy_generation, self._autonomy_enabled = autonomy_generation, autonomy_enabled

    def _autonomy_current(self, generation: int | None) -> bool:
        return generation is None or (self._autonomy_enabled() and generation == self._autonomy_generation())

    def run(self) -> None:
        """
        Starts the main loop for the TTS Synthesizer thread.

        This method continuously checks the TTS input queue for text to synthesize.
        It processes the text, generates speech audio using the TTS model, and puts the audio messages
        into the audio output queue. It handles end-of-stream tokens and logs processing times.
        If an empty or whitespace-only string is received, it logs a warning without processing it.

        The thread will run until the shutdown event is set, at which point it will exit gracefully.
        """
        logger.info("TextToSpeechSynthesizer thread started.")
        generation, autonomy_epoch = self._quiet_generation(), None
        while not self.shutdown_event.is_set():
            try:
                text_to_speak = self.tts_input_queue.get(timeout=self.pause_time)

                generation = getattr(text_to_speak, "generation", None)
                autonomy_epoch = getattr(text_to_speak, "autonomy_generation", None)
                autonomy_cycle = getattr(text_to_speak, "autonomy_cycle", None)
                if generation is None:
                    generation = self._quiet_generation()
                if (self._quiet_mode() or generation != self._quiet_generation()
                        or not self._autonomy_current(autonomy_epoch)):
                    continue
                if text_to_speak == "<EOS>" or (
                    isinstance(text_to_speak, SpeechText) and text_to_speak.text == "<EOS>"
                ):
                    logger.debug("TTS Synthesizer: Received EOS token.")
                    self.audio_output_queue.put(
                        AudioMessage(audio=np.array([], dtype=np.float32), text="", is_eos=True, generation=generation,
                                     autonomy_generation=autonomy_epoch, autonomy_cycle=autonomy_cycle)
                    )

                else:
                    segments = (
                        [text_to_speak]
                        if isinstance(text_to_speak, SpeechText)
                        else SpeechMarkupParser().feed(text_to_speak, final=True)
                    )
                    for segment in segments:
                        if (self._quiet_mode() or generation != self._quiet_generation()
                                or not self._autonomy_current(autonomy_epoch)):
                            break
                        if not segment.text.strip():
                            continue
                        logger.info(f"LLM text: {segment.text}")
                        if self._observability_bus:
                            self._observability_bus.emit(
                                source="tts",
                                kind="synthesize",
                                message=trim_message(segment.text),
                                meta={"generation": generation},
                            )

                        start_time = time.time()
                        spoken_text_variant = self.stc.text_to_spoken(segment.text)
                        if self._tts_muted_event and self._tts_muted_event.is_set():
                            audio_data = np.array([], dtype=np.float32)
                        else:
                            audio_data = self.tts_model.generate_speech_audio(spoken_text_variant)
                        if (self._quiet_mode() or generation != self._quiet_generation()
                                or not self._autonomy_current(autonomy_epoch)):
                            break
                        processing_time = time.time() - start_time

                        audio_duration = len(audio_data) / self.tts_model.sample_rate if audio_data.size else 0.0
                        logger.info(
                            f"TTS Synthesizer: TTS Complete. Inference: {processing_time:.2f}s, "
                            f"Audio length: {audio_duration:.2f}s for text: '{spoken_text_variant}'"
                        )
                        if self._observability_bus:
                            self._observability_bus.emit(
                                source="tts",
                                kind="ready",
                                message=trim_message(spoken_text_variant),
                                level="debug",
                                meta={
                                    "generation": generation,
                                    "inference_s": round(processing_time, 3),
                                    "audio_s": round(audio_duration, 3),
                                    "muted": bool(self._tts_muted_event and self._tts_muted_event.is_set()),
                                },
                            )

                        # Even if audio_data is empty, send the message so AudioPlayer can log/handle it
                        self.audio_output_queue.put(
                            AudioMessage(audio=audio_data, text=spoken_text_variant, emotion=segment.emotion,
                                         generation=generation, autonomy_generation=autonomy_epoch,
                                         autonomy_cycle=autonomy_cycle)
                        )
                        if autonomy_epoch is None:
                            self._on_response_ready(generation, "first_response_ready")
            except queue.Empty:
                pass  # Normal, no text to process
            except Exception as e:
                if autonomy_epoch is None:
                    self._on_response_ready(generation, "synthesis_error")
                logger.exception(f"TextToSpeechSynthesizer: Unexpected error in run loop: {e}")
                # Potentially add a small sleep here
                time.sleep(self.pause_time)

        logger.info("TextToSpeechSynthesizer thread finished.")
