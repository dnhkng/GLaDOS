import queue
import threading
import time

from loguru import logger
import numpy as np

from ..TTS import SpeechSynthesizerProtocol
from ..utils import spoken_text_converter as stc
from .audio_data import AudioMessage


class TextToSpeechSynthesizer:
    """
    A thread that synthesizes text to speech using a TTS model and a spoken text converter.
    It reads text from a queue, processes it, generates audio, and puts the audio messages into an output queue.
    This class is designed to run in a separate thread, continuously checking for new text to
    synthesize until a shutdown event is set.
    """

    def __init__(
        self,
        tts_input_queue: queue.Queue[str],
        audio_output_queue: queue.Queue[AudioMessage],
        tts_model: SpeechSynthesizerProtocol,
        stc_instance: stc.SpokenTextConverter,
        shutdown_event: threading.Event,
        pause_time: float,
    ) -> None:
        self.tts_input_queue = tts_input_queue
        self.audio_output_queue = audio_output_queue
        self.tts_model = tts_model
        self.stc = stc_instance
        self.shutdown_event = shutdown_event
        self.pause_time = pause_time

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
        while not self.shutdown_event.is_set():
            try:
                text_to_speak = self.tts_input_queue.get(timeout=self.pause_time)

                if text_to_speak == "<EOS>":
                    logger.debug("TTS Synthesizer: Received EOS token.")
                    self.audio_output_queue.put(
                        AudioMessage(audio=np.array([], dtype=np.float32), text="", is_eos=True)
                    )

                elif not text_to_speak.strip():  # Check for empty or whitespace-only strings
                    logger.warning(f"TTS Synthesizer: Received empty or whitespace string: '{text_to_speak}'")
                else:
                    logger.info(f"LLM text: {text_to_speak}")

                    start_time = time.time()
                    spoken_text_variant = self.stc.text_to_spoken(text_to_speak)
                    audio_data = self.tts_model.generate_speech_audio(spoken_text_variant)
                    processing_time = time.time() - start_time

                    audio_duration = len(audio_data) / self.tts_model.sample_rate
                    logger.info(
                        f"TTS Synthesizer: TTS Complete. Inference: {processing_time:.2f}s, "
                        f"Audio length: {audio_duration:.2f}s for text: '{spoken_text_variant}'"
                    )

                    # Even if audio_data is empty, send the message so AudioPlayer can log/handle it
                    self.audio_output_queue.put(AudioMessage(audio=audio_data, text=spoken_text_variant, is_eos=False))
            except queue.Empty:
                pass  # Normal, no text to process
            except Exception as e:
                logger.exception(f"TextToSpeechSynthesizer: Unexpected error in run loop: {e}")
                # Potentially add a small sleep here
                time.sleep(self.pause_time)

        logger.info("TextToSpeechSynthesizer thread finished.")
