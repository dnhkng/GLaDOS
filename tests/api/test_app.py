from __future__ import annotations

from litestar.testing import AsyncTestClient
import pytest

pytestmark = pytest.mark.anyio


async def test_create_speech(client: AsyncTestClient) -> None:
    response = await client.post("/v1/audio/speech", json={"input": "The cake is real"})
    assert response.status_code == 201
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.headers["content-disposition"] == 'attachment; filename="speech.mp3"'
    assert isinstance(response.content, bytes)
