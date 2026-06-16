"""ONNX Runtime utility functions for GPU acceleration."""

from __future__ import annotations

import onnxruntime as ort

from .onnx_config import OnnxRuntimeConfig

DEFAULT_MIGRAPHX_OPTIONS: dict[str, str | int] = {
    "device_id": "0",
    "migraphx_fp16_enable": "0",
    "migraphx_bf16_enable": "0",
    "migraphx_exhaustive_tune": "0",
}


def get_provider_priority_list(config: OnnxRuntimeConfig | None = None) -> list[str]:
    """
    Get prioritized list of available execution providers.
    
    Priority order:
    1. MIGraphXExecutionProvider (AMD ROCm - RDNA 3.5/4)
    2. CUDAExecutionProvider (NVIDIA)
    3. CPUExecutionProvider (fallback)
    """
    available = ort.get_available_providers()
    
    filtered = [
        p for p in available 
        if p not in ("TensorrtExecutionProvider", "CoreMLExecutionProvider")
    ]
    
    if config and config.provider_priority:
        providers = []
        for provider in config.provider_priority:
            if provider in filtered:
                providers.append(provider)
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers
    
    providers = []
    
    if "MIGraphXExecutionProvider" in filtered:
        providers.append("MIGraphXExecutionProvider")
    
    if "CUDAExecutionProvider" in filtered:
        providers.append("CUDAExecutionProvider")
    
    providers.append("CPUExecutionProvider")
    
    return providers


def get_migraphx_provider_options(config: OnnxRuntimeConfig) -> list[dict[str, str]]:
    """
    Create provider options for MIGraphX based on configuration.
    
    Args:
        config: ONNX Runtime configuration with MIGraphX settings
        
    Returns:
        List of provider option dictionaries matching the provider order
    """
    migraphx_opts = config.migraphx
    options: dict[str, str] = {
        "device_id": str(migraphx_opts.device_id),
    }
    
    if migraphx_opts.fp16_enable:
        options["migraphx_fp16_enable"] = "1"
    
    if migraphx_opts.bf16_enable:
        options["migraphx_bf16_enable"] = "1"
    
    if migraphx_opts.int8_enable:
        options["migraphx_int8_enable"] = "1"
    
    if migraphx_opts.exhaustive_tune:
        options["migraphx_exhaustive_tune"] = "1"
    
    if migraphx_opts.cache_path:
        options["migraphx_cache_path"] = migraphx_opts.cache_path
    
    if migraphx_opts.mem_limit:
        options["migraphx_mem_limit"] = str(migraphx_opts.mem_limit)
    
    return [options]


def create_session_options(config: OnnxRuntimeConfig | None = None) -> ort.SessionOptions:
    """
    Create ONNX Runtime session options.
    
    Args:
        config: ONNX Runtime configuration
        
    Returns:
        Configured SessionOptions object
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    if config:
        opts.enable_mem_pattern = config.enable_mem_pattern
        if config.intra_op_num_threads is not None:
            opts.intra_op_num_threads = config.intra_op_num_threads
    else:
        opts.enable_mem_pattern = True
    
    return opts
