"""Hardware detection and dynamic model adaptation for GLaDOS.

Continuously and dynamically adapts LLM parameters (context window, generation limits,
concurrency, timeouts, and CPU thread count) to the host machine's architecture,
GPU capabilities (NVIDIA, AMD, Intel Arc, Apple Silicon, or CPU), available memory,
and active model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
from typing import Any
import urllib.request


@dataclass
class HardwareProfile:
    """Detected hardware capabilities, memory budgets and device tiers."""

    ram_total_gb: float = 8.0
    ram_available_gb: float = 4.0
    cpu_count: int = 4
    gpu_type: str = "cpu"  # "nvidia", "intel_arc", "intel_igpu", "amd", "apple_silicon", "cpu"
    gpu_name: str = ""
    vram_total_gb: float | None = None
    vram_free_gb: float | None = None
    unified_memory: bool = False
    tier: str = "medium"  # "ultra", "high", "medium", "low"
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        gpu_str = self.gpu_name or self.gpu_type
        if self.vram_total_gb is not None:
            gpu_str += f" ({self.vram_total_gb:.1f} GB VRAM"
            if self.vram_free_gb is not None:
                gpu_str += f", {self.vram_free_gb:.1f} GB free"
            gpu_str += ")"
        ram = f"{self.ram_total_gb:.1f} GB RAM ({self.ram_available_gb:.1f} GB free)"
        return f"{self.tier.upper()} tier | {gpu_str} | {ram} | {self.cpu_count} CPUs"


@dataclass
class ModelSpec:
    """Estimated or probed metadata for the active language model."""

    name: str = ""
    parameter_size: str = ""  # e.g. "4B", "8B", "14B"
    quantization: str = ""  # e.g. "Q4_K_M", "FP16"
    context_length: int = 32768
    kv_bytes_per_token: int = 147456  # default ~0.14 MB/token (typical for 4B models)


def detect_ram() -> tuple[float, float]:
    """Detect total and available RAM in gigabytes across Linux, macOS and Windows."""
    # 1. psutil if available
    try:
        import psutil

        mem = psutil.virtual_memory()
        return round(mem.total / (1024**3), 2), round(mem.available / (1024**3), 2)
    except (ImportError, OSError):
        pass

    # 2. Linux /proc/meminfo
    if Path("/proc/meminfo").exists():
        try:
            total_b = 0
            avail_b = 0
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_b = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        avail_b = int(line.split()[1]) * 1024
            if total_b > 0:
                avail_b = avail_b or int(total_b * 0.5)
                return round(total_b / (1024**3), 2), round(avail_b / (1024**3), 2)
        except (OSError, ValueError, IndexError):
            pass

    # 3. macOS sysctl
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2, check=True
            ).stdout
            total_b = int(out.strip())
            return round(total_b / (1024**3), 2), round(total_b * 0.5 / (1024**3), 2)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    # 4. Windows ctypes
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024**3), 2), round(stat.ullAvailPhys / (1024**3), 2)
        except (OSError, AttributeError):  # windll only exists on Windows
            pass

    # 5. Fallback via os.sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_b = pages * page_size
        return round(total_b / (1024**3), 2), round(total_b * 0.5 / (1024**3), 2)
    except (OSError, ValueError, AttributeError):  # sysconf names are platform-specific
        return 8.0, 4.0


def detect_gpu() -> tuple[str, str, float | None, float | None, bool]:
    """Detect GPU type, name, total VRAM, free VRAM, and whether unified memory is used."""
    # 1. NVIDIA via nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            out = subprocess.run(
                [nvidia_smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            ).stdout.strip()
            if out:
                first = out.splitlines()[0]
                name, total_mib, free_mib = [x.strip() for x in first.split(",")]
                return "nvidia", name, round(int(total_mib) / 1024, 2), round(int(free_mib) / 1024, 2), False
        except (OSError, subprocess.SubprocessError, ValueError):
            pass

    # 2. Apple Silicon (macOS arm64 unified memory)
    if sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64"):
        try:
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2, check=True
            ).stdout.strip()
            total_ram, avail_ram = detect_ram()
            # macOS Metal can allocate up to 75% of unified memory as VRAM
            vram_total = round(total_ram * 0.75, 2)
            vram_free = round(avail_ram * 0.75, 2)
            return "apple_silicon", brand or "Apple Silicon", vram_total, vram_free, True
        except (OSError, subprocess.SubprocessError):
            return "apple_silicon", "Apple Silicon", None, None, True

    # 3. Linux AMD GPU via sysfs (AMDGPU kernel driver exposes exact VRAM bytes)
    try:
        for drm_dev in Path("/sys/class/drm").glob("card*/device"):
            vram_total_file = drm_dev / "mem_info_vram_total"
            vram_used_file = drm_dev / "mem_info_vram_used"
            if vram_total_file.exists():
                total_b = int(vram_total_file.read_text().strip())
                used_b = int(vram_used_file.read_text().strip()) if vram_used_file.exists() else 0
                free_b = max(0, total_b - used_b)
                total_gb = round(total_b / (1024**3), 2)
                free_gb = round(free_b / (1024**3), 2)
                return "amd", "AMD Radeon / ROCm", total_gb, free_gb, False
    except (OSError, ValueError):
        pass

    # 4. Linux PCI search (Intel Arc, AMD, NVIDIA without SMI)
    lspci = shutil.which("lspci")
    if lspci:
        try:
            pci_out = subprocess.run([lspci], capture_output=True, text=True, timeout=2, check=True).stdout
            for line in pci_out.splitlines():
                lower = line.lower()
                if any(k in lower for k in ("vga compatible controller", "3d controller", "display controller")):
                    desc = line.split(":", 2)[-1].strip()
                    if "intel" in lower:
                        is_arc = any(
                            a in lower
                            for a in (
                                "arc",
                                "lunar lake",
                                "battlemage",
                                "alchemist",
                                "130v",
                                "140v",
                                "a770",
                                "a750",
                                "a580",
                            )
                        )
                        gpu_type = "intel_arc" if is_arc else "intel_igpu"
                        # Intel Arc integrated Lunar Lake shares fast memory
                        total_ram, avail_ram = detect_ram()
                        estimated_vram = round(min(16.0, total_ram * 0.5), 2)
                        return gpu_type, desc, estimated_vram, round(min(avail_ram, estimated_vram), 2), True
                    if "amd" in lower or "radeon" in lower:
                        total_ram, avail_ram = detect_ram()
                        return "amd", desc, round(min(16.0, total_ram * 0.5), 2), round(avail_ram * 0.5, 2), True
        except (OSError, subprocess.SubprocessError):
            pass

    # 5. Linux DRM card nodes fallback
    if Path("/dev/dri").exists() and list(Path("/dev/dri").glob("card*")):
        total_ram, avail_ram = detect_ram()
        return "drm_gpu", "Linux DRM Video Device", round(total_ram * 0.3, 2), round(avail_ram * 0.3, 2), True

    return "cpu", "CPU only", None, None, False


def detect_hardware() -> HardwareProfile:
    """Probe host hardware capabilities and classify into a performance tier."""
    ram_total, ram_avail = detect_ram()
    cpu_count = os.cpu_count() or 4
    gpu_type, gpu_name, vram_total, vram_free, unified = detect_gpu()

    # Determine dynamic performance tier
    tier = "medium"
    if gpu_type == "nvidia" and vram_total is not None and vram_total >= 16.0:
        tier = "ultra"
    elif (gpu_type == "nvidia" and vram_total is not None and vram_total >= 10.0) or (
        gpu_type == "apple_silicon" and ram_total >= 32.0
    ):
        tier = "high"
    elif (
        gpu_type in ("intel_arc", "amd", "apple_silicon")
        or (gpu_type == "nvidia" and (vram_total or 0) >= 6.0)
        or (ram_total >= 16.0 and gpu_type != "cpu")
    ):
        tier = "medium"
    elif gpu_type == "cpu" or ram_total <= 8.0:
        tier = "low"
    else:
        tier = "medium"

    return HardwareProfile(
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        cpu_count=cpu_count,
        gpu_type=gpu_type,
        gpu_name=gpu_name,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
        unified_memory=unified,
        tier=tier,
        details={
            "os": platform.system(),
            "kernel": platform.release(),
            "arch": platform.machine(),
        },
    )


def probe_model_spec(completion_url: str, model_name: str, timeout_s: float = 1.5) -> ModelSpec:
    """Probe language model specifications and compute its KV cache footprint per token."""
    url_lower = completion_url.lower()
    spec = ModelSpec(name=model_name)

    # If Ollama endpoint, query /api/show for real architecture parameters
    if "/api/chat" in url_lower or "/api/generate" in url_lower or ":11434" in url_lower:
        base_url = completion_url.split("/api/")[0]
        show_url = f"{base_url}/api/show"
        try:
            req = urllib.request.Request(
                show_url,
                data=json.dumps({"model": model_name}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                details = data.get("details", {})
                info = data.get("model_info", {})
                spec.parameter_size = details.get("parameter_size", "")
                spec.quantization = details.get("quantization_level", "")
                arch = info.get("general.architecture", "")
                spec.context_length = int(info.get(f"{arch}.context_length", 32768))

                blocks = int(info.get(f"{arch}.block_count", 0))
                heads_kv = int(info.get(f"{arch}.attention.head_count_kv", 0))
                key_len = int(info.get(f"{arch}.attention.key_length", 128))
                val_len = int(info.get(f"{arch}.attention.value_length", 128))

                if blocks and heads_kv:
                    # 2 bytes per fp16 KV entry (key + value)
                    spec.kv_bytes_per_token = blocks * heads_kv * (key_len + val_len) * 2
                    return spec
        except (OSError, ValueError, TypeError, AttributeError):  # URLError is an OSError
            pass

    # Heuristic fallback based on model name
    name_lower = model_name.lower()
    if any(k in name_lower for k in ("1.5b", "2b", "3b")):
        spec.parameter_size = "3B"
        spec.kv_bytes_per_token = 81920  # ~80 KB/token
    elif any(k in name_lower for k in ("4b", "e4b")):
        spec.parameter_size = "4B"
        spec.kv_bytes_per_token = 147456  # ~144 KB/token
    elif any(k in name_lower for k in ("7b", "8b")):
        spec.parameter_size = "8B"
        spec.kv_bytes_per_token = 262144  # ~256 KB/token
    elif "14b" in name_lower:
        spec.parameter_size = "14B"
        spec.kv_bytes_per_token = 524288  # ~512 KB/token
    elif "70b" in name_lower:
        spec.parameter_size = "70B"
        spec.kv_bytes_per_token = 1310720  # ~1.3 MB/token

    return spec


def compute_adaptive_context(profile: HardwareProfile, spec: ModelSpec | None = None) -> int:
    """Calculate the mathematically optimal context window in tokens.

    Ensures KV cache memory comfortably fits within available VRAM / RAM without
    causing GPU out-of-memory or high prompt-eval latency.
    """
    spec = spec or ModelSpec()
    kv_bytes = max(1024, spec.kv_bytes_per_token)

    # Available memory budget dedicated to KV cache
    vram_free = (
        profile.vram_free_gb
        if profile.vram_free_gb is not None
        else (profile.vram_total_gb * 0.7 if profile.vram_total_gb else None)
    )
    if vram_free is not None and not profile.unified_memory:
        # Dedicated VRAM: allocate up to 50% of free VRAM for KV cache (preserving weights and workspace)
        budget_bytes = vram_free * (1024**3) * 0.50
    elif profile.vram_total_gb is not None:
        # Unified memory (Intel Arc, Apple Silicon): cap between 1.0 GB and 3.5 GB
        budget_bytes = min(3.5 * (1024**3), profile.ram_available_gb * (1024**3) * 0.35)
    else:
        # Pure CPU: conservative budget (max 1.5 GB for KV cache)
        budget_bytes = min(1.5 * (1024**3), profile.ram_available_gb * (1024**3) * 0.25)

    max_possible_tokens = int(budget_bytes / kv_bytes)

    # Standard quantized context steps
    standard_steps = (2048, 4096, 8192, 16384, 32768)
    chosen = standard_steps[0]
    for step in standard_steps:
        if step <= max_possible_tokens and step <= spec.context_length:
            chosen = step
        else:
            break

    # Tier-based caps for responsiveness in real-time voice interactions:
    if profile.tier == "low":
        chosen = min(chosen, 4096)
    elif profile.tier == "medium":
        chosen = min(chosen, 8192)
    elif profile.tier == "high":
        chosen = min(chosen, 16384)

    return chosen


def get_adaptive_llm_settings(
    profile: HardwareProfile | None = None,
    spec: ModelSpec | None = None,
    is_local: bool = True,
) -> dict[str, Any]:
    """Dynamically compute tuned LLM options for the host system.

    Parameters:
        profile: Optional pre-computed hardware profile; auto-detected if None.
        spec: Optional probed model specification.
        is_local: True if running against a local endpoint (Ollama, llama.cpp, etc.).

    Returns:
        Dictionary with:
          - 'num_ctx': recommended context length in tokens
          - 'num_predict': maximum generated tokens per voice response clause
          - 'num_thread': optimal CPU thread count
          - 'llm_timeout': maximum request timeout in seconds
          - 'autonomy_parallel_calls': number of background LLM inference workers
          - 'options': nested dict for Ollama API {"num_ctx", "num_predict", "num_thread"}
          - 'hardware_summary': string summary of detected tier and device
          - 'recommendation': guidance string for model selection
    """
    if profile is None:
        profile = detect_hardware()

    if not is_local:
        return {
            "tier": profile.tier,
            "gpu_type": profile.gpu_type,
            "gpu_name": profile.gpu_name,
            "vram_total_gb": profile.vram_total_gb,
            "vram_free_gb": profile.vram_free_gb,
            "ram_total_gb": profile.ram_total_gb,
            "ram_available_gb": profile.ram_available_gb,
            "cpu_count": profile.cpu_count,
            "unified_memory": profile.unified_memory,
            "num_ctx": 16384,
            "num_predict": 1024,
            "num_thread": max(1, min(profile.cpu_count - 1, 8)),
            "llm_timeout": 30.0,
            "autonomy_parallel_calls": 2,
            "options": {"num_ctx": 16384, "num_predict": 1024},
            "hardware_summary": profile.summary(),
            "recommendation": "Remote cloud API active; host hardware compute limits are bypassed.",
        }

    # Dynamically compute context window based on model metadata and memory
    num_ctx = compute_adaptive_context(profile, spec)

    # Optimal CPU thread count (leaves 2 cores for real-time ASR, VAD, TTS)
    num_thread = max(1, min(profile.cpu_count - 2, 8))

    # Output tokens, timeouts and worker lanes per tier
    if profile.tier == "ultra":
        num_predict = 1024
        timeout = 35.0
        workers = 2
        recommendation = "Ultra high-end workstation. Fits 14B-32B models effortlessly with large context."
    elif profile.tier == "high":
        num_predict = 1024
        timeout = 45.0
        workers = 2
        recommendation = (
            "Dedicated GPU / Apple Silicon High Tier. Excellent for 7B-14B models (e.g. Qwen2.5-14B, Llama-3.1-8B)."
        )
    elif profile.tier == "medium":
        num_predict = 512
        timeout = 60.0
        workers = 1  # 1 worker prevents Ollama single-stream (-np 1) slot contention
        recommendation = (
            "Intel Arc / Mid-range GPU. Best suited for 3B-7B models "
            "(e.g. Qwen3-4B-Instruct, Gemma4-E4B) with 8k context."
        )
    else:
        num_predict = 256
        timeout = 90.0
        workers = 1
        recommendation = (
            "CPU or low memory system. Recommend compact models "
            "(1.5B-3B, e.g. Qwen2.5-3B, Gemma2-2B) with 2k-4k context."
        )

    return {
        "tier": profile.tier,
        "gpu_type": profile.gpu_type,
        "gpu_name": profile.gpu_name,
        "vram_total_gb": profile.vram_total_gb,
        "vram_free_gb": profile.vram_free_gb,
        "ram_total_gb": profile.ram_total_gb,
        "ram_available_gb": profile.ram_available_gb,
        "cpu_count": profile.cpu_count,
        "unified_memory": profile.unified_memory,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "num_thread": num_thread,
        "llm_timeout": timeout,
        "autonomy_parallel_calls": workers,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "num_thread": num_thread,
        },
        "hardware_summary": profile.summary(),
        "recommendation": recommendation,
    }


class HardwareAdaptationController:
    """Thread-safe controller that monitors inference latencies and dynamically adjusts settings."""

    def __init__(self, profile: HardwareProfile | None = None) -> None:
        self.profile = profile or detect_hardware()
        self._lock = threading.RLock()
        self._latencies: list[float] = []
        self._current_step_down = 0

    def record_latency(self, seconds: float) -> None:
        """Record real-world time-to-first-token latency."""
        with self._lock:
            self._latencies.append(seconds)
            if len(self._latencies) > 10:
                self._latencies.pop(0)

    def adaptive_context_multiplier(self) -> float:
        """Dynamic multiplier (1.0, 0.5, 0.25) based on observed performance."""
        with self._lock:
            if len(self._latencies) >= 3:
                avg = sum(self._latencies[-3:]) / 3
                if avg > 30.0:
                    return 0.25
                elif avg > 20.0:
                    # Model is struggling on current hardware; step down context size
                    return 0.5
            return 1.0

    def snapshot(self) -> dict[str, Any]:
        """Telemetry snapshot of real-time latency feedback."""
        with self._lock:
            recent = self._latencies[-3:] if len(self._latencies) >= 3 else self._latencies
            recent_avg = round(sum(recent) / len(recent), 2) if recent else None
            return {
                "recent_avg_ttft_s": recent_avg,
                "latency_samples": list(self._latencies),
                "context_multiplier": self.adaptive_context_multiplier(),
            }
