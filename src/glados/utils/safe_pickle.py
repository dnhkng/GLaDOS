"""Load the TTS lookup tables without letting a pickle import or call anything."""

from pathlib import Path
import pickle
from typing import Any, NoReturn


class _DataOnlyUnpickler(pickle.Unpickler):
    # Dicts, lists, strings and numbers need no globals; any class reference means a tampered file.
    def find_class(self, module: str, name: str) -> NoReturn:
        raise pickle.UnpicklingError(f"Refusing to load {module}.{name} from a data-only pickle")


def load_data_pickle(path: Path) -> dict[Any, Any]:
    with path.open("rb") as f:
        data = _DataOnlyUnpickler(f).load()
    if not isinstance(data, dict):
        raise pickle.UnpicklingError(f"{path} does not contain a dictionary")
    return data
