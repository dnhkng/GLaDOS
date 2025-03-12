# This file is for autodiscovery by the litestar CLI,
# so we don't need to pass an app every time or use an environment variable.
from glados.api import app

__all__ = ["app"]
