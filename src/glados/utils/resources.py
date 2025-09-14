from functools import lru_cache
import os
from pathlib import Path

import glados


@lru_cache(maxsize=1)
def get_package_root() -> Path:
    """Get the absolute path to the package root directory (cached)."""
    
    # Get the directory where the glados module is located
    package_dir = Path(os.path.dirname(os.path.abspath(glados.__file__)))

    # Go up to the project root (src/glados -> src -> project_root)
    package_dir = package_dir.parent.parent

    if os.getenv("PYAPP") == "1":
        package_dir = package_dir / "site-packages/glados"

    return package_dir


def resource_path(relative_path: str) -> Path:
    """Return absolute path to a model file."""
    if os.getenv("PYAPP") == "1":
        root_path = Path(os.getenv("PYAPP_RELATIVE_DIR")) # Relative path to the executable
    else:
        root_path = get_package_root()

    return root_path / relative_path
