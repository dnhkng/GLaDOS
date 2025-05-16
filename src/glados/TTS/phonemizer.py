# ruff: noqa: RUF001
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from pickle import load
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray
import onnxruntime as ort  # type: ignore

from ..utils.resources import resource_path

# Default OnnxRuntime is way to verbose, only show fatal errors
ort.set_default_logger_severity(4)


@dataclass
class ModelConfig:
    MODEL_PATH: Path
    PHONEME_DICT_PATH: Path
    TOKEN_TO_IDX_PATH: Path
    IDX_TO_TOKEN_PATH: Path
    CHAR_REPEATS: int = 3
    MODEL_INPUT_LENGTH: int = 64
    EXPAND_ACRONYMS: bool = False
    USE_CUDA: bool = True

    def __init__(
        self,
        model_path: Path | None = None,
        phoneme_dict_path: Path | None = None,
        token_to_idx_path: Path | None = None,
        idx_to_token_path: Path | None = None,
    ) -> None:
        # Provide default Path objects if None is passed or for defaults
        self.MODEL_PATH = model_path if model_path is not None else resource_path("models/TTS/phomenizer_en.onnx")
        self.PHONEME_DICT_PATH = (
            phoneme_dict_path if phoneme_dict_path is not None else resource_path("models/TTS/lang_phoneme_dict.pkl")
        )
        self.TOKEN_TO_IDX_PATH = (
            token_to_idx_path if token_to_idx_path is not None else resource_path("models/TTS/token_to_idx.pkl")
        )
        self.IDX_TO_TOKEN_PATH = (
            idx_to_token_path if idx_to_token_path is not None else resource_path("models/TTS/idx_to_token.pkl")
        )


class SpecialTokens(Enum):
    PAD = "_"
    START = "<start>"
    END = "<end>"
    EN_US = "<en_us>"


class Punctuation(Enum):
    PUNCTUATION = "().,:?!/–"
    HYPHEN = "-"
    SPACE = " "

    @classmethod
    @cache
    def get_punc_set(cls) -> set[str]:
        return set(cls.PUNCTUATION.value + cls.HYPHEN.value + cls.SPACE.value)

    @classmethod
    @cache
    def get_punc_pattern(cls) -> re.Pattern[str]:
        """
        Compile a regular expression pattern to match punctuation and space characters.

        Returns:
            re.Pattern[str]: A compiled regex pattern that matches any punctuation or space character.

        Example:
            pattern = Punctuation.get_punc_pattern()
            # Matches single punctuation or space characters
            matches = pattern.findall("Hello, world!")  # Returns [',', ' ']
        """
        return re.compile(f"([{cls.PUNCTUATION.value + cls.SPACE.value}])")


class Phonemizer:
    """Phonemizer class for converting text to phonemes.

    This class uses an ONNX model to predict phonemes for a given text.

    Attributes:
        phoneme_dict: Dictionary of phonemes for each word.
        token_to_idx: Mapping of tokens to indices.
        idx_to_token: Mapping of indices to tokens.
        ort_session: ONNX runtime session for the model.
        special_tokens: Set of special tokens.

    Methods:
        _load_pickle(path: Path) -> dict:
            Load a pickled dictionary from path.

        _unique_consecutive(arr: np.ndarray) -> List[np.ndarray]:
            Equivalent to torch.unique_consecutive for numpy arrays.

        _remove_padding(arr: np.ndarray, padding_value: int = 0) -> List[np.ndarray]:
            Remove padding from an array.

        _trim_to_stop(arr: np.ndarray, end_index: int = 2) -> List[np.ndarray]:
            Trim an array to the stop index.

        _process_model_output(arr: np.ndarray) -> List[np.ndarray]:
            Process the output of the model to get the phoneme indices.

        _expand_acronym(word: str) -> str:
            Expand an acronym into its subwords.

        encode(sentence: Iterable[str]) -> List[int]:
            Map a sequence of symbols to a sequence of indices.

        decode(sequence: np.ndarray) -> str:
            Map a sequence of indices to a sequence of symbols.

        _pad_sequence_fixed(v: List[np.ndarray], target_length: int = ModelConfig.MODEL_INPUT_LENGTH) -> np.ndarray:
            Pad or truncate a list of arrays to a fixed length.

        _get_dict_entry(word: str, lang: str, punc_set: set[str]) -> str | None:
            Get the phoneme entry for a word in the dictionary.

        _get_phonemes(word: str, word_phonemes: Dict[str, Union[str, None]], word_splits: Dict[str, List[str]]) -> str:
            Get the phonemes for a word.

        _clean_and_split_texts(
            texts: List[str],
            punc_set: set[str],
            punc_pattern: re.Pattern
        ) -> tuple[List[List[str]], set[str]]:
            Clean and split texts.

        convert_to_phonemes(texts: List[str], lang: str) -> List[str]:
            Convert a list of texts to phonemes using a phonemizer.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """
        Initialize a Phonemizer instance with optional configuration.

        Parameters:
            config (ModelConfig, optional): Configuration settings for the phonemizer.
                If not provided, a default ModelConfig will be used.

        Attributes:
            config (ModelConfig): Configuration for the phonemizer.
            phoneme_dict (dict[str, str]): Dictionary mapping words to their phonetic representations.
            token_to_idx (dict): Mapping of tokens to their corresponding indices.
            idx_to_token (dict): Mapping of indices back to tokens.
            ort_session (InferenceSession): ONNX runtime session for model inference.
            special_tokens (set[str]): Set of special tokens used in phonemization.

        Notes:
            - Adds a special phoneme entry for "glados"
            - Configures ONNX runtime session with available providers, excluding TensorRT
        """
        if config is None:
            config = ModelConfig()
        self.config = config
        self.phoneme_dict: dict[str, str] = self._load_pickle(self.config.PHONEME_DICT_PATH)

        self.phoneme_dict["glados"] = "ɡlˈɑːdɑːs"  # Add GLaDOS to the phoneme dictionary!

        self.token_to_idx = self._load_pickle(self.config.TOKEN_TO_IDX_PATH)
        self.idx_to_token = self._load_pickle(self.config.IDX_TO_TOKEN_PATH)

        providers = ort.get_available_providers()
        if "TensorrtExecutionProvider" in providers:
            providers.remove("TensorrtExecutionProvider")
        if "CoreMLExecutionProvider" in providers:
            providers.remove("CoreMLExecutionProvider")

        self.ort_session = ort.InferenceSession(
            self.config.MODEL_PATH,
            sess_options=ort.SessionOptions(),
            providers=providers,
        )

        self.special_tokens: set[str] = {
            SpecialTokens.PAD.value,
            SpecialTokens.END.value,
            SpecialTokens.EN_US.value,
        }

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        """
        Load a pickled dictionary from the specified file path.

        Parameters:
            path (Path): The file path to the pickled dictionary file.

        Returns:
            dict[str, Any]: The loaded dictionary containing key-value pairs.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            pickle.UnpicklingError: If there are issues unpickling the file.
        """
        with path.open("rb") as f:
            return load(f)  # type: ignore

    @staticmethod
    def _unique_consecutive(arr: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
        """
        Remove consecutive duplicate elements from each array in the input list.

        This method is analogous to PyTorch's `unique_consecutive` function, but implemented for NumPy arrays.
        It filters out repeated adjacent elements, keeping only the first occurrence of consecutive duplicates.

        Args:
            arr (list[NDArray[np.int64]]): A list of NumPy integer arrays to process.

        Returns:
            list[NDArray[np.int64]]: A list of arrays with consecutive duplicates removed.

        Example:
            Input: [[1, 1, 2, 2, 3, 3, 3]]
            Output: [[1, 2, 3]]
        """

        result = []
        for row in arr:
            if len(row) == 0:
                result.append(row)
            else:
                mask = np.concatenate(([True], row[1:] != row[:-1]))
                result.append(row[mask])

        return result

    @staticmethod
    def _remove_padding(arr: list[NDArray[np.int64]], padding_value: int = 0) -> list[NDArray[np.int64]]:
        """
        Remove padding values from input arrays.

        Parameters:
            arr (list[NDArray[np.int64]]): A list of numpy arrays containing integer values
            padding_value (int, optional): The value to be considered as padding. Defaults to 0.

        Returns:
            list[NDArray[np.int64]]: A list of numpy arrays with padding values removed
        """
        return [row[row != padding_value] for row in arr]

    @staticmethod
    def _trim_to_stop(arr: list[NDArray[np.int64]], end_index: int = 2) -> list[NDArray[np.int64]]:
        """
        Trims each input array to the first occurrence of a specified stop index.

        This method searches for the first occurrence of the specified end index in each input array
        and truncates the array up to and including that index. If no such index is found, the original
        array is returned unchanged.

        Parameters:
            arr (list[NDArray[np.int64]]): List of numpy integer arrays to be trimmed.
            end_index (int, optional): The index value used as a stopping point. Defaults to 2.

        Returns:
            list[NDArray[np.int64]]: A list of trimmed numpy arrays, where each array is cut off
            at the first occurrence of the end_index (inclusive).

        Example:
            Input: [[1, 2, 3, 4], [5, 2, 6, 7]]
            With end_index=2
            Output: [[1, 2], [5, 2]]
        """
        result = []
        for row in arr:
            stop_index = np.where(row == end_index)[0]
            if len(stop_index) > 0:
                result.append(row[: stop_index[0] + 1])
            else:
                result.append(row)
        return result

    def _process_model_output(self, arr: list[NDArray[np.int64]]) -> list[NDArray[np.int64]]:
        """
        Process the ONNX model's output to extract phoneme indices with post-processing.

        This method transforms raw model output into a clean sequence of phoneme indices by applying
        several filtering techniques:
        1. Converts model probabilities to index selections using argmax
        2. Removes consecutive duplicate indices
        3. Removes padding tokens
        4. Trims the sequence to the stop token

        Args:
            arr (list[NDArray[np.int64]]): Raw model output containing phoneme probability distributions.

        Returns:
            list[NDArray[np.int64]]: Processed phoneme indices with duplicates, padding, and excess tokens removed.
        """

        arr_processed: list[NDArray[np.int64]] = np.argmax(arr[0], axis=2)
        arr_processed = self._unique_consecutive(arr_processed)
        arr_processed = self._remove_padding(arr_processed)
        arr_processed = self._trim_to_stop(arr_processed)

        return arr_processed

    @staticmethod
    def _expand_acronym(word: str) -> str:
        """
        Expands an acronym into its subwords, with current implementation preserving the original acronym.

        This method handles acronyms by maintaining their original form. Currently, it performs two key checks:
        - If the word contains a hyphen, it returns the word unchanged
        - For other acronyms, it returns the word as-is

        Parameters:
            word (str): The input word potentially representing an acronym

        Returns:
            str: The processed word, which remains unchanged for acronyms

        Notes:
            - Designed to work with true acronyms
            - Mixed case acronym handling is delegated to SpokenTextConverter
        """
        # Only split on hyphens if they exist
        if Punctuation.HYPHEN.value in word:
            return word

        # For acronyms, just return as is - they're already preprocessed
        return word

    def encode(self, sentence: Iterable[str]) -> list[int]:
        """
        Converts a sequence of symbols into a sequence of token indices with special start and end tokens.

        This method prepares input for phoneme conversion by:
        1. Repeating each input symbol based on the configured character repeat setting
        2. Converting symbols to lowercase
        3. Mapping symbols to their corresponding token indices
        4. Adding special start and end tokens to the sequence

        Parameters:
            sentence (Iterable[str]): A sequence of symbols (characters or words) to be encoded.

        Returns:
            list[int]: A sequence of token indices representing the input, including start and end special tokens.

        Example:
            # Assuming token_to_idx maps 'h' to 10, 'i' to 15, with START at 1 and END at 2
            phonemizer.encode(['hi'])  # Returns [1, 10, 15, 2]
        """
        sentence = [item for item in sentence for _ in range(self.config.CHAR_REPEATS)]
        sentence = [s.lower() for s in sentence]
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        return [
            self.token_to_idx[SpecialTokens.START.value],
            *sequence,
            self.token_to_idx[SpecialTokens.END.value],
        ]

    def decode(self, sequence: NDArray[np.int64]) -> str:
        """
        Converts a sequence of token indices back to a human-readable string of symbols.

        This method decodes a numpy array of integer indices into their corresponding token representations,
        filtering out special tokens to produce a clean output string.

        Args:
            sequence (NDArray[np.int64]): A numpy array of integer indices representing encoded tokens.

        Returns:
            str: A decoded string containing only meaningful tokens, with special tokens removed.

        Example:
            # Assuming self.idx_to_token maps indices to tokens
            decoded_text = phonemizer.decode(np.array([5, 10, 3, 1]))  # Returns a string of decoded tokens
        """
        decoded = []

        for t in sequence:
            idx = t.item()
            token = self.idx_to_token[idx]
            decoded.append(token)

        result = "".join(d for d in decoded if d not in self.special_tokens)
        return result

    @staticmethod
    def pad_sequence_fixed(v: list[list[int]], target_length: int) -> NDArray[np.int64]:
        """
        Pad or truncate a list of integer sequences to a fixed length.

        This method ensures all input sequences have a uniform length by either:
        - Truncating sequences longer than the target length
        - Padding sequences shorter than the target length with zeros

        Parameters:
            v (list[list[int]]): A list of integer sequences to be padded/truncated
            target_length (int): The desired uniform length for all sequences

        Returns:
            NDArray[np.int64]: A 2D numpy array with sequences of uniform length,
            where each row represents a padded/truncated sequence
        """

        result: NDArray[np.int64] = np.zeros((len(v), target_length), dtype=np.int64)

        for i, seq in enumerate(v):
            length = min(len(seq), target_length)  # Handle both shorter and longer sequences
            result[i, :length] = seq[:length]  # Copy either the full sequence or its truncated version

        return result

    def _get_dict_entry(self, word: str, punc_set: set[str]) -> str | None:
        """
        Retrieves the phoneme entry for a given word from the phoneme dictionary.

        This method handles different word variations by checking the dictionary with original, lowercase, and
        title-cased versions of the word. It also handles punctuation and empty strings as special cases.

        Args:
            word (str): The word to look up in the phoneme dictionary.
            punc_set (set[str]): A set of punctuation characters.

        Returns:
            str | None: The phoneme entry for the word if found, the word itself if it's a punctuation or
            empty string, or None if no entry exists.
        """
        if word in punc_set or len(word) == 0:
            return word
        if word in self.phoneme_dict:
            return self.phoneme_dict[word]

        elif word.lower() in self.phoneme_dict:
            return self.phoneme_dict[word.lower()]

        elif word.title() in self.phoneme_dict:
            return self.phoneme_dict[word.title()]
        else:
            return None

    @staticmethod
    def _get_phonemes(
        word: str,
        word_phonemes: dict[str, str | None],
        word_splits: dict[str, list[str]],
    ) -> str:
        """
        Get the phonemes for a given word, handling dictionary lookup and subword processing.

        Parameters:
            word (str): The word to retrieve phonemes for.
            word_phonemes (dict[str, str | None]): A dictionary mapping words to their phoneme representations.
            word_splits (dict[str, list[str]]): A dictionary mapping words to their subword splits.

        Returns:
            str: The phoneme representation of the word. If the word is not directly in the dictionary,
                 it attempts to construct phonemes from its subwords, filtering out any None values.

        Raises:
            KeyError: If the word is not found in either the word_phonemes or word_splits dictionaries.
        """
        phons = word_phonemes[word]
        if phons is None:
            subwords = word_splits[word]
            subphons_converted = [word_phonemes[w] for w in subwords]
            phons = "".join([subphon for subphon in subphons_converted if subphon is not None])
        return phons

    def _clean_and_split_texts(
        self, texts: list[str], punc_set: set[str], punc_pattern: re.Pattern[str]
    ) -> tuple[list[list[str]], set[str]]:
        """
        Clean and split input texts into words while preserving specified punctuation.

        This method performs text preprocessing by removing non-alphanumeric characters (except specified punctuation),
        splitting the text into words, and collecting unique cleaned words.

        Parameters:
            texts (list[str]): List of input text strings to be cleaned and split.
            punc_set (set[str]): Set of punctuation characters to preserve during cleaning.
            punc_pattern (re.Pattern[str]): Regular expression pattern for splitting text.

        Returns:
            tuple[list[list[str]], set[str]]: A tuple containing:
                - A list of lists, where each inner list represents words from a corresponding input text
                - A set of unique cleaned words across all input texts
        """
        split_text, cleaned_words = [], set[str]()
        for text in texts:
            cleaned_text = "".join(t for t in text if t.isalnum() or t in punc_set)
            split = [s for s in re.split(punc_pattern, cleaned_text) if len(s) > 0]
            split_text.append(split)
            cleaned_words.update(split)
        return split_text, cleaned_words

    def convert_to_phonemes(self, texts: list[str], lang: str = "en_us") -> list[str]:
        """
        Converts a list of texts to phonemes using a phonemizer.

        This method processes input texts through several stages:
        1. Preprocess and clean input texts
        2. Collect phonemes from an existing dictionary
        3. Split words that are not in the dictionary
        4. Predict phonemes for missing words using an ONNX model
        5. Reconstruct phonemes for each input text

        Parameters:
            texts (list[str]): A list of text strings to convert to phonemes.
            lang (str, optional): Language of the texts. Defaults to "en_us".

        Returns:
            list[str]: A list of phoneme representations corresponding to the input texts.

        Notes:
            - Handles punctuation and special characters
            - Supports acronym expansion
            - Uses a pre-trained ONNX model for phoneme prediction
            - Supports multiple input texts simultaneously

        Example:
            phonemizer = Phonemizer()
            texts = ["Hello world", "OpenAI"]
            phonemes = phonemizer.convert_to_phonemes(texts)
            # Possible output: ["HH AH0 L OW1 W ER0 L D", "OW1 P AH0 N EY1"]
        """
        split_text: list[list[str]] = []
        cleaned_words = set[str]()

        punc_set = Punctuation.get_punc_set()
        punc_pattern = Punctuation.get_punc_pattern()

        # Step 1: Preprocess texts
        split_text, cleaned_words = self._clean_and_split_texts(texts, punc_set, punc_pattern)

        # Step 2: Collect dictionary phonemes for words and hyphenated words
        for punct in punc_set:
            self.phoneme_dict[punct] = punct
        word_phonemes = {word: self.phoneme_dict.get(word.lower()) for word in cleaned_words}

        # Step 3: If word is not in dictionary, split it into subwords
        words_to_split = [w for w in cleaned_words if word_phonemes[w] is None]

        word_splits = {
            key: re.split(
                r"([-])",
                self._expand_acronym(word) if self.config.EXPAND_ACRONYMS else word,
            )
            for key, word in zip(words_to_split, words_to_split, strict=False)
        }

        subwords = {w for values in word_splits.values() for w in values if w not in word_phonemes}

        for subword in subwords:
            word_phonemes[subword] = self._get_dict_entry(word=subword, punc_set=punc_set)

        # Step 4: Predict all subwords that are missing in the phoneme dict
        words_to_predict = [
            word for word, phons in word_phonemes.items() if phons is None and len(word_splits.get(word, [])) <= 1
        ]

        if words_to_predict:
            input_batch = [self.encode(word) for word in words_to_predict]
            input_batch_padded: NDArray[np.int64] = self.pad_sequence_fixed(input_batch, self.config.MODEL_INPUT_LENGTH)

            ort_inputs = {self.ort_session.get_inputs()[0].name: input_batch_padded}
            ort_outs = self.ort_session.run(None, ort_inputs)

            ids = self._process_model_output(ort_outs)

            # Step 5: Add predictions to the dictionary
            for id, word in zip(ids, words_to_predict, strict=False):
                word_phonemes[word] = self.decode(id)

        # Step 6: Get phonemes for each word in the text
        phoneme_lists = []
        for text in split_text:
            text_phons = [
                self._get_phonemes(word=word, word_phonemes=word_phonemes, word_splits=word_splits) for word in text
            ]
            phoneme_lists.append(text_phons)

        return ["".join(phoneme_list) for phoneme_list in phoneme_lists]

    def __del__(self) -> None:
        """Clean up ONNX session to prevent context leaks."""
        if hasattr(self, "ort_sess"):
            del self.ort_sess
