"""Tests for hardware profiling and adaptive model configuration."""

from __future__ import annotations

from unittest.mock import patch

from glados.core.engine import GladosConfig
from glados.core.hardware import (
    HardwareProfile,
    detect_gpu,
    detect_hardware,
    detect_ram,
    get_adaptive_llm_settings,
)


def test_detect_ram_returns_valid_values() -> None:
    total, avail = detect_ram()
    assert total > 0
    assert avail > 0
    assert avail <= total


def test_detect_gpu_runs_without_crashing() -> None:
    gpu_type, gpu_name, vram_total, _vram_free, unified = detect_gpu()
    assert isinstance(gpu_type, str)
    assert isinstance(gpu_name, str)
    assert vram_total is None or vram_total >= 0
    assert isinstance(unified, bool)


def test_detect_hardware_profiles_system() -> None:
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.tier in ("ultra", "high", "medium", "low")
    assert profile.cpu_count >= 1
    assert profile.ram_total_gb > 0
    summary = profile.summary()
    assert profile.tier.upper() in summary


def test_adaptive_settings_for_different_tiers() -> None:
    high_prof = HardwareProfile(ram_total_gb=64, tier="high", gpu_type="nvidia", vram_total_gb=16)
    med_prof = HardwareProfile(ram_total_gb=32, tier="medium", gpu_type="intel_arc", vram_total_gb=8)
    low_prof = HardwareProfile(ram_total_gb=8, tier="low", gpu_type="cpu")

    high_opts = get_adaptive_llm_settings(high_prof, is_local=True)
    assert high_opts["num_ctx"] == 16384
    assert high_opts["autonomy_parallel_calls"] == 2
    assert high_opts["llm_timeout"] == 45.0
    assert high_opts["tier"] == "high"

    med_opts = get_adaptive_llm_settings(med_prof, is_local=True)
    assert med_opts["num_ctx"] == 8192
    assert med_opts["autonomy_parallel_calls"] == 1
    assert med_opts["llm_timeout"] == 60.0
    assert med_opts["tier"] == "medium"

    low_opts = get_adaptive_llm_settings(low_prof, is_local=True)
    assert low_opts["num_ctx"] == 4096
    assert low_opts["autonomy_parallel_calls"] == 1
    assert low_opts["llm_timeout"] == 90.0
    assert low_opts["tier"] == "low"

    remote_opts = get_adaptive_llm_settings(low_prof, is_local=False)
    assert remote_opts["llm_timeout"] == 30.0


def test_glados_config_applies_hardware_adaptation() -> None:
    cfg = GladosConfig.from_yaml("configs/glados_extended_config.yaml")
    assert cfg.adaptive_hardware is True
    assert cfg.llm_timeout is not None
    assert "options" in (cfg.llm_request_options or {})
    assert "num_ctx" in cfg.llm_request_options["options"]
    assert cfg.llm_request_options["options"]["num_ctx"] in (4096, 8192, 16384)


def test_glados_config_preserves_explicit_overrides() -> None:
    with patch.dict("os.environ", {}, clear=True):
        cfg = GladosConfig.from_yaml(["configs/glados_extended_config.yaml"])
        # Explicit override
        custom_cfg = GladosConfig.model_validate(
            {
                **cfg.model_dump(),
                "llm_timeout": 120.0,
                "llm_request_options": {"options": {"num_ctx": 2048}},
            }
        )
        assert custom_cfg.llm_timeout == 120.0
        assert custom_cfg.llm_request_options["options"]["num_ctx"] == 2048


def test_hardware_adaptation_controller_adjusts_context() -> None:
    from glados.core.hardware import HardwareAdaptationController

    ctrl = HardwareAdaptationController()
    assert ctrl.adaptive_context_multiplier() == 1.0

    # Normal latencies
    ctrl.record_latency(2.5)
    ctrl.record_latency(3.1)
    ctrl.record_latency(2.8)
    assert ctrl.adaptive_context_multiplier() == 1.0

    # High latencies (>20s) trigger step down
    ctrl.record_latency(22.0)
    ctrl.record_latency(25.0)
    ctrl.record_latency(21.0)
    assert ctrl.adaptive_context_multiplier() == 0.5

    # Very high latencies (>30s) trigger further step down
    ctrl.record_latency(35.0)
    ctrl.record_latency(32.0)
    ctrl.record_latency(31.0)
    assert ctrl.adaptive_context_multiplier() == 0.25

    snap = ctrl.snapshot()
    assert snap["recent_avg_ttft_s"] > 30.0
    assert snap["context_multiplier"] == 0.25
