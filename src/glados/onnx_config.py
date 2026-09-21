"""Configuration for ONNX Runtime execution providers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MIGraphXOptions(BaseModel):
    """MIGraphX Execution Provider configuration options."""
    
    device_id: int = Field(default=0, ge=0, description="GPU device ID")
    fp16_enable: bool = Field(default=False, description="Enable FP16 quantization")
    bf16_enable: bool = Field(default=False, description="Enable BF16 quantization")
    int8_enable: bool = Field(default=False, description="Enable INT8 quantization")
    exhaustive_tune: bool = Field(default=False, description="Enable exhaustive kernel tuning")
    cache_path: str | None = Field(default=None, description="Model compilation cache path")
    mem_limit: int | None = Field(default=None, description="Memory arena limit")


class OnnxRuntimeConfig(BaseModel):
    """ONNX Runtime configuration."""
    
    provider_priority: list[str] | None = Field(
        default=None,
        description="Execution provider priority list. Auto-detected if None."
    )
    
    migraphx: MIGraphXOptions = Field(
        default_factory=MIGraphXOptions,
        description="MIGraphX Execution Provider options"
    )
    
    enable_mem_pattern: bool = Field(
        default=True,
        description="Enable memory pattern optimization"
    )
    intra_op_num_threads: int | None = Field(
        default=None,
        description="Number of threads for intra-op parallelism"
    )
