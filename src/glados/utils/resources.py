from functools import lru_cache
import os
from pathlib import Path

import glados


@lru_cache(maxsize=1)
def get_package_root() -> Path:
    """Get the absolute path to the package root directory (cached)."""
    # Get the directory where the glados module is located
    package_dir = Path(os.path.dirname(os.path.abspath(glados.__file__)))

    # During pyapp runtime, the python package glados is located here:
    # /home/yourUsername/.local/share/pyapp/glados/12151318069254528956/0.1.0/lib/python3.12/site-packages/glados
    if os.getenv("PYAPP") != "1":
        # Go up to the project root (src/glados -> src -> project_root)
        package_dir = package_dir.parent.parent

    return package_dir


def resource_path(relative_path: str) -> Path:
    """Return absolute path to a model file."""
    if os.getenv("PYAPP") == "1":
        root_path = Path(os.getenv("PYAPP_RELATIVE_DIR")) # Relative path to the executable
    else:
        root_path = get_package_root()

    return root_path / relative_path
