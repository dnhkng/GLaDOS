"""FastVLM ONNX-based vision-language model for scene description."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import onnxruntime as ort
import regex as re
from loguru import logger
from numpy.typing import NDArray

from ..utils.resources import resource_path

# Suppress ONNX verbose logging
ort.set_default_logger_severity(4)

_DEFAULT_SYSTEM_PROMPT: Final[str] = "You are a helpful assistant."


def _bytes_to_unicode() -> dict[int, str]:
    """Create a reversible byte-to-unicode mapping."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Return set of symbol pairs in a word."""
    return set(zip(word, word[1:], strict=False))


class _ByteBPETokenizer:
    """Minimal ByteLevel BPE tokenizer for Qwen2-based FastVLM."""

    def __init__(self, tokenizer_path: Path, tokenizer_config_path: Path) -> None:
        tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        config_data = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))

        self.vocab: dict[str, int] = tokenizer_data["model"]["vocab"]
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        merges = tokenizer_data["model"]["merges"]
        self.bpe_ranks = {}
        for i, merge in enumerate(merges):
            if isinstance(merge, str):
                pair = tuple(merge.split())
            else:
                pair = tuple(merge)
            self.bpe_ranks[pair] = i

        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache: dict[str, str] = {}

        pattern = tokenizer_data["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        self.pattern = re.compile(pattern)

        self.special_tokens: dict[str, int] = {}
        added_tokens = config_data.get("added_tokens_decoder", {})
        if added_tokens:
            self.special_tokens = {value["content"]: int(key) for key, value in added_tokens.items()}
        else:
            for token in tokenizer_data.get("added_tokens", []):
                self.special_tokens[token["content"]] = token["id"]

        self.special_token_ids = set(self.special_tokens.values())
        self._special_sorted = sorted(self.special_tokens.keys(), key=len, reverse=True)

    def encode(self, text: str) -> list[int]:
        """Encode a string into token IDs."""
        token_ids: list[int] = []
        for part in self._split_on_special(text):
            if not part:
                continue
            if part in self.special_tokens:
                token_ids.append(self.special_tokens[part])
                continue

            for token in self.pattern.findall(part):
                token_bytes = token.encode("utf-8")
                token_str = "".join(self.byte_encoder[b] for b in token_bytes)
                for bpe_token in self._bpe(token_str).split(" "):
                    token_id = self.vocab.get(bpe_token)
                    if token_id is not None:
                        token_ids.append(token_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to a string."""
        tokens: list[str] = []
        for token_id in token_ids:
            if token_id in self.special_token_ids:
                continue
            token = self.id_to_token.get(token_id)
            if token:
                tokens.append(token)

        text = "".join(tokens)
        byte_values = [self.byte_decoder[c] for c in text]
        return bytes(byte_values).decode("utf-8", errors="replace").strip()

    def _split_on_special(self, text: str) -> list[str]:
        if not self._special_sorted:
            return [text]

        parts: list[str] = []
        buffer = []
        i = 0
        while i < len(text):
            matched = False
            for token in self._special_sorted:
                if text.startswith(token, i):
                    if buffer:
                        parts.append("".join(buffer))
                        buffer = []
                    parts.append(token)
                    i += len(token)
                    matched = True
                    break
            if not matched:
                buffer.append(text[i])
                i += 1
        if buffer:
            parts.append("".join(buffer))
        return parts

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                if j < len(word) - 1 and word[j] == first and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str


class FastVLM:
    """ONNX-based vision-language model for scene description."""

    DEFAULT_MODEL_DIR = resource_path("models/Vision/FastVLM")

    def __init__(self, model_dir: Path | None = None) -> None:
        """Initialize FastVLM with ONNX models.

        Args:
            model_dir: Path to directory containing ONNX models and config files.
                      Uses DEFAULT_MODEL_DIR if None.
        """
        model_dir = model_dir or self.DEFAULT_MODEL_DIR
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)

        logger.info(f"Loading FastVLM from {model_dir}")

        # Configure providers (same pattern as ASR)
        providers = ort.get_available_providers()
        for excluded in ["TensorrtExecutionProvider", "CoreMLExecutionProvider"]:
            if excluded in providers:
                providers.remove(excluded)

        if "CUDAExecutionProvider" in providers:
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self._providers = ["CPUExecutionProvider"]

        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_opts.enable_mem_pattern = True

        onnx_dir = model_dir / "onnx"

        logger.debug("Loading vision encoder...")
        self.vision_encoder = ort.InferenceSession(
            str(onnx_dir / "vision_encoder_q4.onnx"),
            sess_options=session_opts,
            providers=self._providers,
        )

        logger.debug("Loading embed tokens...")
        self.embed_tokens = ort.InferenceSession(
            str(onnx_dir / "embed_tokens_fp16.onnx"),
            sess_options=session_opts,
            providers=self._providers,
        )

        logger.debug("Loading decoder...")
        self.decoder = ort.InferenceSession(
            str(onnx_dir / "decoder_model_merged_q4.onnx"),
            sess_options=session_opts,
            providers=self._providers,
        )

        self._decoder_layers = self._count_decoder_layers()
        self._past_num_heads, self._past_head_dim = self._get_past_shape()

        self._load_configs(model_dir)

        logger.success(f"FastVLM loaded using {self._providers[0]}")

    def _load_configs(self, model_dir: Path) -> None:
        """Load tokenizer and preprocessing configs."""
        self.config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        preprocessor = json.loads((model_dir / "preprocessor_config.json").read_text(encoding="utf-8"))

        tokenizer_path = model_dir / "tokenizer.json"
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        self._tokenizer = _ByteBPETokenizer(tokenizer_path, tokenizer_config_path)

        self.image_token_id = int(self.config.get("image_token_index", self._tokenizer.special_tokens.get("<image>", -1)))
        self.eos_token_id = int(self.config.get("eos_token_id", self._tokenizer.special_tokens.get("<|im_end|>", -1)))
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"

        crop_size = preprocessor.get("crop_size", {})
        self._image_size = int(crop_size.get("height", 1024))
        self._rescale_factor = float(preprocessor.get("rescale_factor", 1.0 / 255.0))
        self._do_center_crop = bool(preprocessor.get("do_center_crop", True))
        self._do_resize = bool(preprocessor.get("do_resize", True))

    def _count_decoder_layers(self) -> int:
        """Count the number of decoder layers based on past_key_values inputs."""
        past_keys = [
            inp for inp in self.decoder.get_inputs() if inp.name.startswith("past_key_values.") and inp.name.endswith(".key")
        ]
        return max(1, len(past_keys))

    def _get_past_shape(self) -> tuple[int, int]:
        """Infer the past key/value head and dim sizes from the decoder inputs."""
        for inp in self.decoder.get_inputs():
            if inp.name == "past_key_values.0.key":
                shape = inp.shape
                num_heads = int(shape[1]) if isinstance(shape[1], int) else 2
                head_dim = int(shape[3]) if isinstance(shape[3], int) else 64
                return num_heads, head_dim
        return 2, 64

    def describe(self, image: NDArray[np.uint8], prompt: str, max_tokens: int = 64) -> str:
        """Generate a description of the image.

        Args:
            image: Input image in BGR format (OpenCV default), uint8, HWC layout
            prompt: Prompt describing what to generate
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text description
        """
        prompt = prompt.strip() if prompt else "Describe the image."

        pixel_values = self._preprocess_image(image)

        vision_outputs = self.vision_encoder.run(None, {"pixel_values": pixel_values})
        vision_features = vision_outputs[0]

        prompt_text = self._apply_chat_template(f"<image>{prompt}")
        prompt_token_ids = self._tokenizer.encode(prompt_text)

        if self.image_token_id not in prompt_token_ids:
            logger.warning("FastVLM: <image> token missing from prompt; description may be degraded.")

        generated_ids = self._generate(vision_features, prompt_token_ids, max_tokens)
        generated_text = self._tokenizer.decode(generated_ids[len(prompt_token_ids):])
        return generated_text

    def _apply_chat_template(self, content: str) -> str:
        """Apply the Qwen2 chat template for a single user message."""
        return (
            f"{self.im_start_token}system\n{_DEFAULT_SYSTEM_PROMPT}{self.im_end_token}\n"
            f"{self.im_start_token}user\n{content}{self.im_end_token}\n"
            f"{self.im_start_token}assistant\n"
        )

    def _preprocess_image(self, image: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Resize, normalize, convert to model input format."""
        resized = image
        if self._do_resize:
            height, width = image.shape[:2]
            scale = self._image_size / float(min(height, width))
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        if self._do_center_crop:
            height, width = resized.shape[:2]
            top = max(0, (height - self._image_size) // 2)
            left = max(0, (width - self._image_size) // 2)
            resized = resized[top : top + self._image_size, left : left + self._image_size]

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) * self._rescale_factor

        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _generate(
        self,
        vision_features: NDArray[np.float32],
        prompt_token_ids: list[int],
        max_tokens: int,
    ) -> list[int]:
        """Autoregressive text generation."""
        generated_ids = prompt_token_ids.copy()

        input_ids = np.array([generated_ids], dtype=np.int64)
        embeds_outputs = self.embed_tokens.run(None, {"input_ids": input_ids})
        inputs_embeds = embeds_outputs[0]

        try:
            image_token_pos = generated_ids.index(self.image_token_id)
        except ValueError:
            image_token_pos = -1

        if image_token_pos >= 0:
            before = inputs_embeds[:, :image_token_pos, :]
            after = inputs_embeds[:, image_token_pos + 1 :, :]
            inputs_embeds = np.concatenate([before, vision_features, after], axis=1)
        else:
            inputs_embeds = np.concatenate([inputs_embeds, vision_features], axis=1)

        attention_mask = np.ones((1, inputs_embeds.shape[1]), dtype=np.int64)
        past_key_values = [
            (
                np.zeros((1, self._past_num_heads, 0, self._past_head_dim), dtype=np.float32),
                np.zeros((1, self._past_num_heads, 0, self._past_head_dim), dtype=np.float32),
            )
            for _ in range(self._decoder_layers)
        ]

        for _ in range(max_tokens):
            decoder_inputs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
            }

            if inputs_embeds.shape[1] == attention_mask.shape[1]:
                position_ids = np.arange(attention_mask.shape[1], dtype=np.int64)[None, :]
            else:
                position_ids = np.array([[attention_mask.shape[1] - 1]], dtype=np.int64)
            decoder_inputs["position_ids"] = position_ids

            for i, (key, value) in enumerate(past_key_values):
                decoder_inputs[f"past_key_values.{i}.key"] = key
                decoder_inputs[f"past_key_values.{i}.value"] = value

            decoder_outputs = self.decoder.run(None, decoder_inputs)

            logits = decoder_outputs[0]
            past_key_values = [
                (decoder_outputs[i * 2 + 1], decoder_outputs[i * 2 + 2])
                for i in range((len(decoder_outputs) - 1) // 2)
            ]

            next_token_logits = logits[:, -1, :]
            next_token_id = int(np.argmax(next_token_logits, axis=-1)[0])

            if next_token_id == self.eos_token_id:
                break

            generated_ids.append(next_token_id)

            next_input_ids = np.array([[next_token_id]], dtype=np.int64)
            embeds_outputs = self.embed_tokens.run(None, {"input_ids": next_input_ids})
            inputs_embeds = embeds_outputs[0]

            attention_mask = np.concatenate(
                [attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1
            )

        return generated_ids

    def __del__(self) -> None:
        """Clean up ONNX sessions."""
        for attr in ["vision_encoder", "decoder", "embed_tokens"]:
            if hasattr(self, attr):
                delattr(self, attr)
