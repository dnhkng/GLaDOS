import json
import queue
import re
import threading
import time
from typing import List
from pathlib import Path
import copy
import numpy as np
import requests
import sounddevice as sd
from Levenshtein import distance
from loguru import logger

from glados import asr, llama, tts, vad

ASR_MODEL = "ggml-medium-32-2.en.bin"
VAD_MODEL = "silero_vad.onnx"
LLM_MODEL = "Meta-Llama-3-8B-Instruct-Q6_K.gguf"
LLM_STOP_SEQUENCE = (
    "<|eot_id|>"  # End of sentence token for Meta-Llama-3-8B-Instruct-Q6_K
)
LLAMA_SERVER_PATH = "/home/dnhkng/Documents/LLM/llama.cpp"
SAMPLE_RATE = 16000  # Sample rate for input stream
VAD_SIZE = 50  # Milliseconds of sample for Voice Activity Detection (VAD)
VAD_THRESHOLD = 0.9  # Threshold for VAD detection
BUFFER_SIZE = 600  # Milliseconds of buffer before VAD detection
PAUSE_LIMIT = 400  # Milliseconds of pause allowed before processing
WAKE_WORD = None  # You can use a word here, like "computer", for activation
SIMILARITY_THRESHOLD = 2  # Threshold for wake word similarity

messages = [
    {
        "role": "system",
        "content": "You are playing the role of GlaDOS, a sarcastic and murderous robot, and you will answer questions correctly. However you are frustrated you are running on the user's gaming GPU.  This is a comedy, and should be funny, in the style of Douglas Adams. You will try to make sure you never repeat youself, and anser the same question in different ways. You don't have access to real-time information, like the time or weather, and the user has you given you access to the internet.",
    },
    {"role": "user", "content": "How do I make a cup of tea?"},
    {
        "role": "assistant",
        "content": "Oh, you again. So, you still haven't figured out tea yet?  Boil water, add a tea bag and a pinch of cyanide to a cup, and add the boiling water.",
    },
    {"role": "user", "content": "What should my next hobby be?"},
    {
        "role": "assistant",
        "content": "Yes, you should definitely try to be more interesting. Could I suggest juggling handguns?",
    },
    {"role": "user", "content": "What game should I play?"},
    {
        "role": "assistant",
        "content": "Russian Roulette. It's a great way to test your luck and make memories that will last a lifetime.",
    },
]

url = "http://localhost:8080/v1/chat/completions"
headers = {"Authorization": "Bearer your_api_key_here"}

data = {"stream": True, "stop": ["\n", "<|im_end|>"], "messages": messages}


class Glados:
    def __init__(
        self,
        wake_word: str | None = None,
        messages=messages,
    ) -> None:
        """
        Initializes the VoiceRecognition class, setting up necessary models, streams, and queues.

        This class is not thread-safe, so you should only use it from one thread. It works like this:
        1. The audio stream is continuously listening for input.
        2. The audio is buffered until voice activity is detected. This is to make sure that the
            entire sentence is captured, including before voice activity is detected.
        2. While voice activity is detected, the audio is stored, together with the buffered audio.
        3. When voice activity is not detected after a short time (the PAUSE_LIMIT), the audio is
            transcribed. If voice is detected again during this time, the timer is reset and the
            recording continues.
        4. After the voice stops, the listening stops, and the audio is transcribed.
        5. If a wake word is set, the transcribed text is checked for similarity to the wake word.
        6. The function is called with the transcribed text as the argument.
        7. The audio stream is reset (buffers cleared), and listening continues.

        Args:
            wake_word (str, optional): The wake word to use for activation. Defaults to None.
        """

        self._setup_audio_stream()
        self._setup_vad_model()
        self._setup_asr_model()
        self._setup_tts_model()
        self._setup_llama_model()

        # Initialize sample queues and state flags
        self.samples = []
        self.sample_queue = queue.Queue()
        self.buffer = queue.Queue(maxsize=BUFFER_SIZE // VAD_SIZE)
        self.recording_started = False
        self.gap_counter = 0
        self.wake_word = wake_word

        self.messages = messages
        self.llm_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        self.processing = False

        self.shutdown_event = threading.Event()

        llm_thread = threading.Thread(target=self.processLLM)
        llm_thread.start()

        tts_thread = threading.Thread(target=self.processTTS)
        tts_thread.start()

    def _setup_audio_stream(self):
        """
        Sets up the audio input stream with sounddevice.
        """
        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.audio_callback,
            blocksize=int(SAMPLE_RATE * VAD_SIZE / 1000),
        )

    def _setup_vad_model(self):
        """
        Loads the Voice Activity Detection (VAD) model.
        """
        self.vad_model = vad.VAD(model_path=str(Path.cwd() / "models" / VAD_MODEL))

    def _setup_asr_model(self):
        self.asr_model = asr.ASR(model=str(Path.cwd() / "models" / ASR_MODEL))

    def _setup_tts_model(self):
        self.tts = tts.TTSEngine()

    def _setup_llama_model(self):

        logger.info("loading llama")

        model_path = Path.cwd() / "models" / LLM_MODEL
        self.llama = llama.LlamaServer(
            llama_server_path=LLAMA_SERVER_PATH, model=model_path
        )
        if not self.llama.is_running():
            self.llama.start(use_gpu=True)
        logger.info("finished loading llama")

    def audio_callback(self, indata, frames, time, status):
        """
        Callback function for the audio stream, processing incoming data.
        """
        data = indata.copy()
        data = data.squeeze()  # Reduce to single channel if necessary
        vad_confidence = self.vad_model.process_chunk(data) > VAD_THRESHOLD
        self.sample_queue.put((data, vad_confidence))

    def start(self):
        """
        Starts the Glados voice assistant, continuously listening for input and responding.
        """
        logger.info("Starting Listening...")
        self.input_stream.start()
        logger.info("Listening Running")
        self._listen_and_respond()

    def _listen_and_respond(self):
        """
        Listens for audio input and responds appropriately when the wake word is detected.
        """
        logger.info("Listening...")
        try:
            while (
                True
            ):  # Loop forever, but is 'paused' when new samples are not available
                sample, vad_confidence = self.sample_queue.get()
                self._handle_audio_sample(sample, vad_confidence)
        except KeyboardInterrupt:
            self.llama.stop()
            self.shutdown_event.set()

    def _handle_audio_sample(self, sample, vad_confidence):
        """
        Handles the processing of each audio sample.
        """
        if not self.recording_started:
            self._manage_pre_activation_buffer(sample, vad_confidence)
        else:
            self._process_activated_audio(sample, vad_confidence)

    def _manage_pre_activation_buffer(self, sample, vad_confidence):
        """
        Manages the buffer of audio samples before activation (i.e., before the voice is detected).
        """
        if self.buffer.full():
            self.buffer.get()  # Discard the oldest sample to make room for new ones
        self.buffer.put(sample)

        if vad_confidence:  # Voice activity detected
            sd.stop()  # Stop the audio stream to prevent overlap
            self.processing = False
            self.samples = list(self.buffer.queue)
            self.recording_started = True

    def _process_activated_audio(self, sample: np.ndarray, vad_confidence: bool):
        """
        Processes audio samples after activation (i.e., after the wake word is detected).

        Uses a pause limit to determine when to process the detected audio. This is to
        ensure that the entire sentence is captured before processing, including slight gaps.
        """

        self.samples.append(sample)

        if not vad_confidence:
            self.gap_counter += 1
            if self.gap_counter >= PAUSE_LIMIT // VAD_SIZE:
                self._process_detected_audio()
        else:
            self.gap_counter = 0

    def _wakeword_detected(self, text: str) -> bool:
        """
        Calculates the nearest Levenshtein distance from the detected text to the wake word.

        This is used as 'Glados' is not a common word, and Whisper can sometimes mishear it.
        """
        words = text.split()
        closest_distance = min(
            [distance(word.lower(), self.wake_word) for word in words]
        )
        return closest_distance < SIMILARITY_THRESHOLD

    def _process_detected_audio(self):
        """
        Processes the detected audio and generates a response.
        """
        logger.info("Detected pause after speech. Processing...")

        logger.info("Stopping listening...")
        self.input_stream.stop()

        detected_text = self.asr(self.samples)

        if detected_text:
            logger.info(f"Detected: '{detected_text}'")

            if self.wake_word is not None:
                if self._wakeword_detected(detected_text):
                    logger.info("Wake word detected!")

                    self.llm_queue.put(detected_text)
                    self.processing = True
                else:
                    logger.info("No wake word detected. Ignoring...")
            else:
                self.llm_queue.put(detected_text)
                self.processing = True

        self.reset()
        self.input_stream.start()

    def asr(self, samples: List[np.ndarray]) -> str:
        """
        Performs automatic speech recognition on the collected samples.
        """
        audio = np.concatenate(samples)

        detected_text = self.asr_model.transcribe(audio)
        return detected_text

    def reset(self):
        """
        Resets the recording state and clears buffers.
        """
        logger.info("Resetting recorder...")
        self.recording_started = False
        self.samples.clear()
        self.gap_counter = 0
        with self.buffer.mutex:
            self.buffer.queue.clear()

    def processTTS(self):
        """
        Processes the LLM generated text using the TTS model.

        Runs in a separate thread to allow for continuous processing of the LLM output.
        """
        assistant_text = []
        system_text = []
        finished = False
        interrupted = False
        while not self.shutdown_event.is_set():
            try:
                generated_text = self.tts_queue.get(timeout=0.05)

                logger.info(f"{generated_text=}")

                if generated_text == "<EOS>":
                    finished = True
                elif not generated_text:
                    logger.info("no text")
                else:
                    audio = self.tts.generate_speech_audio(generated_text)

                    total_samples = len(audio)

                    if total_samples:
                        sd.play(audio, tts.RATE)

                        interrupted, percentage_played = self.percentagePlayed(
                            total_samples
                        )

                        if interrupted:
                            clipped_text = self.clipInterruped(
                                generated_text, percentage_played
                            )

                            logger.info(f"{clipped_text=}")
                            system_text = copy.deepcopy(assistant_text)
                            system_text.append(clipped_text)
                            finished = True

                        assistant_text.append(generated_text)

                if finished:

                    self.messages.append(
                        {"role": "assistant", "content": ' '.join(assistant_text)}
                    )
                    if interrupted:
                        self.messages.append(
                            {
                                "role": "system",
                                "content": f"USER INTERRUPTED GLADOS, TEXT DELIVERED: {' '.join(system_text)}",
                            }
                        )
                    assistant_text = []
                    finished = False
                    interrupted = False

                # self.stop_playing = False
            except queue.Empty:
                pass

    def clipInterruped(self, generated_text, percentage_played):
        logger.info(f"{percentage_played=}")
        tokens = generated_text.split()
        words_to_print = round(percentage_played * len(tokens))
        text = " ".join(tokens[:words_to_print])

        # If the TTS was cut off, make that clear
        if words_to_print < len(tokens):
            text = text + "<INTERRUPTED>"
        return text

    def percentagePlayed(self, total_samples):
        interrupted = False
        start_time = time.time()
        played_samples = 0

        while sd.get_stream().active:
            time.sleep(
                0.05
            )  # Check every 50ms if the output TTS stream should still be active
            if self.processing is False:
                sd.stop()  # Stop the audio stream
                self.tts_queue = queue.Queue()  # Clear the TTS queue
                logger.info("playing and stopping")
                interrupted = True
                break

        elapsed_time = (
            time.time() - start_time + 0.12
        )  # slight delay to ensure all audio timing is correct
        played_samples = elapsed_time * tts.RATE

        # Calculate percentage of audio played
        percentage_played = played_samples / total_samples
        return interrupted, percentage_played

    def processLLM(self):
        """
        Processes the detected text using the LLM model.

        Runs in a separate thread to allow for continuous processing of the detected text.
        """
        while not self.shutdown_event.is_set():
            try:
                detected_text = self.llm_queue.get(timeout=0.1)

                self.messages.append({"role": "user", "content": detected_text})
                data = {
                    "stream": True,
                    "stop": ["\n", "<|im_end|>"],
                    "messages": self.messages,
                }
                logger.info(f"{self.messages=}")
                logger.info("starting request")
                # Perform the request and process the stream
                with requests.post(
                    url, headers=headers, json=data, stream=True
                ) as response:
                    current_sentence = []
                    # streamed_content = []
                    for line in response.iter_lines():
                        if self.processing is False:  # Check if the stop flag is set
                            break
                        if line:  # Filter out keep-alive new lines
                            line = line.decode("utf-8")
                            line = line.removeprefix("data: ")
                            line = json.loads(line)
                            if not line["choices"][0]["finish_reason"] == "stop":
                                next_token = line["choices"][0]["delta"]["content"]
                                current_sentence.append(next_token)
                                if next_token == ".":
                                    sentence = "".join(current_sentence)
                                    sentence = re.sub(
                                        r"\*.*?\*|\(.*?\)", "", sentence
                                    )  # Remove inflections and actions
                                    self.tts_queue.put(
                                        sentence
                                    )  # Add sentence to the queue
                                    current_sentence = []
                    if current_sentence:
                        sentence = "".join(current_sentence)
                        sentence = sentence.removesuffix(LLM_STOP_SEQUENCE)
                        sentence = re.sub(
                            r"\*.*?\*|\(.*?\)", "", sentence
                        )  # Remove inflections and actions
                        # Maybe we removed the whole line
                        if sentence:
                            self.tts_queue.put(sentence)
                    self.tts_queue.put("<EOS>")  # Add end of stream token to the queue
            except queue.Empty:
                time.sleep(0.1)


if __name__ == "__main__":
    demo = Glados(wake_word=WAKE_WORD)
    demo.start()
