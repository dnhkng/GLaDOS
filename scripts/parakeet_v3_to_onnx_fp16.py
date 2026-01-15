#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import tempfile
from typing import Any

import nemo.collections.asr as nemo_asr
import onnx
from omegaconf import OmegaConf
from onnxconverter_common import float16


DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PREFIX = "parakeet-tdt-0.6b-v3"


def load_or_download_model(model_name: str, nemo_path: Path | None) -> nemo_asr.models.ASRModel:
    if nemo_path and nemo_path.exists():
        logging.info("Restoring model from %s", nemo_path)
        return nemo_asr.models.ASRModel.restore_from(str(nemo_path))

    logging.info("Downloading model %s", model_name)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    if nemo_path:
        logging.info("Saving .nemo file to %s", nemo_path)
        model.save_to(str(nemo_path))
    return model


def save_config(model: nemo_asr.models.ASRModel, output_dir: Path, prefix: str) -> Path:
    config_path = output_dir / f"{prefix}_model_config.yaml"
    OmegaConf.save(model._cfg, config_path)
    logging.info("Saved model config to %s", config_path)
    return config_path


def fix_fp16_casts(model: onnx.ModelProto) -> int:
    fixed_count = 0
    float_type = 1
    float16_type = 10

    graph_output_names = {out.name for out in model.graph.output}

    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        feeds_output = any(out_name in graph_output_names for out_name in node.output)
        if feeds_output:
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == float_type:
                attr.i = float16_type
                fixed_count += 1

    inputs_to_fix = [inp for inp in model.graph.input if inp.type.tensor_type.elem_type == float16_type]
    if inputs_to_fix:
        for inp in inputs_to_fix:
            original_name = inp.name
            cast_output_name = f"{original_name}_fp16_internal"

            inp.type.tensor_type.elem_type = float_type

            cast_node = onnx.helper.make_node(
                "Cast",
                inputs=[original_name],
                outputs=[cast_output_name],
                to=float16_type,
                name=f"cast_input_{original_name}",
            )

            for node in model.graph.node:
                for i, input_name in enumerate(node.input):
                    if input_name == original_name:
                        node.input[i] = cast_output_name

            model.graph.node.insert(0, cast_node)
            fixed_count += 1

    return fixed_count


def add_metadata(model: onnx.ModelProto, meta_data: dict[str, Any]) -> None:
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)


def convert_to_fp16(input_path: Path, output_path: Path, meta_data: dict[str, Any]) -> None:
    logging.info("Loading ONNX %s", input_path)
    model = onnx.load(str(input_path), load_external_data=True)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
    )
    fixed = fix_fp16_casts(model_fp16)
    add_metadata(model_fp16, meta_data)
    logging.info("Saving FP16 ONNX to %s (fixed casts: %s)", output_path, fixed)
    onnx.save(model_fp16, str(output_path))


def export_onnx(model: nemo_asr.models.ASRModel, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    normalize_type = model.cfg.preprocessor.normalize
    if normalize_type == "NA":
        normalize_type = ""

    meta_data = {
        "vocab_size": model.decoder.vocab_size,
        "normalize_type": normalize_type,
        "pred_rnn_layers": model.decoder.pred_rnn_layers,
        "pred_hidden": model.decoder.pred_hidden,
        "subsampling_factor": 8,
        "model_type": "EncDecRNNTBPEModel",
        "version": "3",
        "model_author": "NeMo",
        "url": "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
        "comment": "Exported to FP16 ONNX",
        "feat_dim": 128,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        encoder_path = tmp_path / "encoder.onnx"
        decoder_path = tmp_path / "decoder.onnx"
        joiner_path = tmp_path / "joiner.onnx"

        logging.info("Exporting encoder")
        model.encoder.export(str(encoder_path))
        logging.info("Exporting decoder")
        model.decoder.export(str(decoder_path))
        logging.info("Exporting joiner")
        model.joint.export(str(joiner_path))

        convert_to_fp16(
            encoder_path,
            output_dir / f"{prefix}_encoder.onnx",
            meta_data,
        )
        convert_to_fp16(
            decoder_path,
            output_dir / f"{prefix}_decoder.onnx",
            meta_data,
        )
        convert_to_fp16(
            joiner_path,
            output_dir / f"{prefix}_joiner.onnx",
            meta_data,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Parakeet v3 to FP16 ONNX.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="NeMo model name to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/ASR"),
        help="Output directory for ONNX files and config.",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Filename prefix for exported artifacts.",
    )
    parser.add_argument(
        "--nemo-path",
        type=Path,
        default=None,
        help="Optional path to save the .nemo file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    model = load_or_download_model(args.model_name, args.nemo_path)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(model, args.output_dir, args.prefix)
    export_onnx(model, args.output_dir, args.prefix)

    logging.info("Export complete.")


if __name__ == "__main__":
    main()
