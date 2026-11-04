import importlib.util

# The HTTP API is an optional extra (`uv sync --extra api`); skip its tests when it is not installed.
collect_ignore = [] if importlib.util.find_spec("litestar") else ["api", "test_api_config.py"]
