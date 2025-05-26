# --- glados_orchestrator.py (or rename your main glados.py) ---
from pathlib import Path  # For config loading
import queue
import sys
import threading
import time

from loguru import logger
import sounddevice as sd  # type: ignore

from glados.ASR import VAD, get_audio_transcriber  # Adjust path

# Your new component classes
from .audio_input_handler import AudioInputHandler
from .audio_player import AudioPlayer

# Config and other utilities
from .glados_config import AudioMessage, GladosConfig  # PersonalityPrompt if still used directly by orchestrator
from .llm_processor import LanguageModelProcessor
from .TTS import tts_glados, tts_kokoro
from .tts_synthesizer import TextToSpeechSynthesizer
from .utils import spoken_text_converter as stc  # Adjust path

# Configure logger (can be done once at the top level)
# logger.remove(0)
# logger.add(sys.stderr, level="INFO") # Or DEBUG for more verbose output

class GladosOrchestrator:
    # Keep constants if they are specific to orchestration or default fallbacks
    PAUSE_TIME_GENERAL: float = 0.05 # General pause for threads if not component-specific

    def __init__(self, config: GladosConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        
        # This event signals that an LLM/TTS response cycle is actively being processed.
        # - AudioInputHandler CLEARS it on interruption.
        # - AudioInputHandler SETS it when valid user speech (post-wake-word) is sent to LLM.
        # - LLMProcessor CHECKS it to stop streaming.
        # - AudioPlayer CHECKS it (via its callback) to stop playback.
        # - AudioPlayer CLEARS currently_speaking_event when TTS stream (EOS) is fully done OR on interruption.
        self.processing_active_event = threading.Event()
        self.processing_active_event.set() # Initially, not processing a specific user command that could be interrupted

        # This event signals that the TTS is currently outputting sound.
        # - AudioPlayer SETS it before playing a segment.
        # - AudioPlayer CLEARS it when EOS is processed or playback is interrupted.
        # - AudioInputHandler CHECKS it to know if an interruption is happening.
        self.currently_speaking_event = threading.Event()

        self.conversation_history: list[dict[str, str]] = list(config.to_chat_messages())

        # --- Queues ---
        self.llm_input_queue: queue.Queue[str] = queue.Queue()
        self.tts_input_queue: queue.Queue[str] = queue.Queue() # For text chunks to TTS
        self.audio_output_queue: queue.Queue[AudioMessage] = queue.Queue() # For AudioMessage objects

        # --- Initialize Models & Utilities (moved from old Glados.from_config) ---
        logger.info("Orchestrator: Initializing VAD model...")
        self.vad_model = VAD()
        
        logger.info(f"Orchestrator: Initializing ASR engine: {config.asr_engine}...")
        self.asr_model = get_audio_transcriber(engine_type=config.asr_engine)
        
        logger.info(f"Orchestrator: Initializing TTS voice: {config.voice}...")
        if config.voice == "glados":
            self.tts_model = tts_glados.Synthesizer()
        else:
            # Ensure tts_kokoro.get_voices() is accessible or handle voice validation differently
            # assert config.voice in tts_kokoro.get_voices(), f"Voice '{config.voice}' not available"
            self.tts_model = tts_kokoro.Synthesizer(voice=config.voice)
        
        self.stc_instance = stc.SpokenTextConverter()

        # --- Initialize Components ---
        logger.info("Orchestrator: Initializing AudioInputHandler...")
        self.audio_input_handler = AudioInputHandler(
            llm_input_queue=self.llm_input_queue,
            vad_model=self.vad_model,
            asr_model=self.asr_model,
            config=config, # Pass the whole config; component will pick what it needs
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event
        )

        logger.info("Orchestrator: Initializing LanguageModelProcessor...")
        self.llm_processor = LanguageModelProcessor(
            llm_input_queue=self.llm_input_queue,
            tts_input_queue=self.tts_input_queue,
            conversation_history=self.conversation_history, # Shared
            completion_url=config.completion_url,
            model_name=config.model,
            api_key=config.api_key,
            processing_active_event=self.processing_active_event,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME_GENERAL
        )

        logger.info("Orchestrator: Initializing TextToSpeechSynthesizer...")
        self.tts_synthesizer = TextToSpeechSynthesizer(
            tts_input_queue=self.tts_input_queue,
            audio_output_queue=self.audio_output_queue,
            tts_model=self.tts_model,
            stc_instance=self.stc_instance,
            shutdown_event=self.shutdown_event,
            pause_time=self.PAUSE_TIME_GENERAL
        )

        logger.info("Orchestrator: Initializing AudioPlayer...")
        self.audio_player = AudioPlayer(
            audio_output_queue=self.audio_output_queue,
            conversation_history=self.conversation_history, # Shared
            tts_sample_rate=self.tts_model.sample_rate,
            interruptible=config.interruptible,
            shutdown_event=self.shutdown_event,
            currently_speaking_event=self.currently_speaking_event,
            processing_active_event=self.processing_active_event,
            pause_time=self.PAUSE_TIME_GENERAL
        )
        
        self.component_threads: list[threading.Thread] = []

    def _play_announcement(self) -> None:
        if self.config.announcement:
            logger.info(f"Orchestrator: Playing announcement: '{self.config.announcement}'")
            try:
                # Temporarily set processing_active for announcement if it's not interruptible by default
                # However, announcements usually shouldn't be interruptible by subsequent system logic
                # but might be by immediate user speech if interruptible is globally true.
                # For simplicity, we'll let the AudioPlayer handle it based on the global interruptible flag.
                
                self.processing_active_event.set() # Ensure it's set so player doesn't think it's interrupted
                self.currently_speaking_event.set() # We are about to speak

                spoken_text = self.stc_instance.text_to_spoken(self.config.announcement)
                audio = self.tts_model.generate_speech_audio(spoken_text)
                
                if len(audio) > 0:
                    # Use the AudioPlayer's own playback mechanism for consistency
                    # This is a bit of a direct call, ideally one might queue it.
                    # But for a one-off announcement, this direct control is okay.
                    interrupted, _ = self.audio_player._percentage_played_refactored(audio)
                    if interrupted:
                        logger.warning("Orchestrator: Announcement was interrupted.")
                else:
                    logger.warning("Orchestrator: Announcement TTS produced no audio.")
                
                self.currently_speaking_event.clear() # Announcement done
                # processing_active_event remains set, as the system is now ready for user input
                
            except Exception as e:
                logger.exception(f"Orchestrator: Error playing announcement: {e}")


    def start_services(self) -> None:
        """Starts all component threads."""
        logger.info("Orchestrator: Starting component services...")
        
        # Play announcement before starting listener if needed, or concurrently
        # If announcement should not be interruptible by listening, play it first.
        self._play_announcement()


        thread_targets = {
            "AudioInput": self.audio_input_handler.run,
            "LLMProcessor": self.llm_processor.run,
            "TTSSynthesizer": self.tts_synthesizer.run,
            "AudioPlayer": self.audio_player.run,
        }

        for name, target_func in thread_targets.items():
            thread = threading.Thread(target=target_func, name=name, daemon=True)
            self.component_threads.append(thread)
            thread.start()
            logger.info(f"Orchestrator: {name} thread started.")
        
        logger.success("Orchestrator: All services started.")

    def stop_services(self) -> None:
        """Stops all component threads gracefully."""
        if self.shutdown_event.is_set():
            logger.info("Orchestrator: Shutdown already in progress.")
            return

        logger.info("Orchestrator: Initiating service shutdown...")
        self.shutdown_event.set() # Signal all threads to stop

        # Clear events to help unblock threads waiting on them, if applicable
        self.processing_active_event.set() # Allow any waiting logic to proceed to shutdown check
        self.currently_speaking_event.clear()


        for thread in self.component_threads:
            logger.debug(f"Orchestrator: Joining {thread.name} thread...")
            thread.join(timeout=5) # Wait for threads to finish
            if thread.is_alive():
                logger.warning(f"Orchestrator: Thread {thread.name} did not shut down cleanly after 5s.")
        
        # Explicitly close sounddevice streams if they weren't closed by threads
        # (AudioInputHandler and AudioPlayer should handle their own stream closures)
        sd.stop(ignore_errors=True) # Stop any remaining sounddevice activity as a final measure

        logger.success("Orchestrator: All services stopped.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GladosOrchestrator":
        """Loads config and creates an orchestrator instance."""
        # This method can replace the old Glados.from_yaml
        logger.info(f"Orchestrator: Loading configuration from {path}...")
        glados_config = GladosConfig.from_yaml(path)
        return cls(glados_config)


# --- Main execution block (e.g., in your main script) ---
def main() -> None:
    # Setup logger (do this once)
    logger.remove(0)
    # logger.add(sys.stderr, level="DEBUG") # For detailed debugging
    logger.add(sys.stderr, level="INFO") # For standard run
    # logger.add(sys.stderr, level="SUCCESS") # If you only want success and errors

    try:
        orchestrator = GladosOrchestrator.from_yaml("glados_config.yaml")
        orchestrator.start_services()

        # Keep the main thread alive, listening for KeyboardInterrupt
        while not orchestrator.shutdown_event.is_set():
            time.sleep(0.5) # Check periodically

    except KeyboardInterrupt:
        logger.info("MAIN: KeyboardInterrupt received. Shutting down orchestrator...")
    except FileNotFoundError:
        logger.error("MAIN: Configuration file 'glados_config.yaml' not found.")
    except Exception as e:
        logger.exception(f"MAIN: An unexpected error occurred: {e}")
    finally:
        if 'orchestrator' in locals() and orchestrator:
            orchestrator.stop_services()
        logger.info("MAIN: Application terminated.")

if __name__ == "__main__":
    main()