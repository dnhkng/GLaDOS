from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_package_root() -> Path:
    """Get the absolute path to the project root directory (cached)."""
    # utils -> glados -> src -> project_root
    return Path(__file__).resolve().parents[3]


def resource_path(relative_path: str) -> Path:
    """Return absolute path to a model file."""
    return get_package_root() / relative_path
