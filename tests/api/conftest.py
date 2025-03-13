from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from litestar.testing import AsyncTestClient
import pytest

from glados.api.app import app

if TYPE_CHECKING:
    from litestar import Litestar

app.debug = True


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="function")
async def client() -> AsyncIterator[AsyncTestClient[Litestar]]:
    async with AsyncTestClient(app=app) as client:
        yield client
