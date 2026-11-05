"""glados download: plain files and models shipped inside an archive, both checksum-verified."""

import asyncio
from hashlib import sha256
import io
from pathlib import Path
import tarfile

import httpx
from rich.progress import Progress

from glados.cli import download_model


def _archive(member: str, data: bytes) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo(member)
        info.size = len(data)
        bundle.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _fetch(path: Path, info: dict[str, str], body: bytes) -> bool:
    async def run() -> bool:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body))
        async with httpx.AsyncClient(transport=transport) as client:
            with Progress(disable=True) as progress:
                return await download_model(client, path, info, progress)

    return asyncio.run(run())


def test_a_model_is_unpacked_from_its_archive(tmp_path: Path) -> None:
    model = b"onnx bytes"
    archive = _archive("./voice.onnx", model)
    info = {
        "url": "https://example.test/voice.tar.gz",
        "archive_member": "./voice.onnx",
        "archive_checksum": sha256(archive).hexdigest(),
        "checksum": sha256(model).hexdigest(),
    }
    target = tmp_path / "voice.onnx"
    assert _fetch(target, info, archive)
    assert target.read_bytes() == model
    assert sorted(p.name for p in tmp_path.iterdir()) == ["voice.onnx"], "the archive is removed"


def test_a_wrong_model_checksum_writes_nothing(tmp_path: Path) -> None:
    archive = _archive("./voice.onnx", b"tampered")
    info = {
        "url": "https://example.test/voice.tar.gz",
        "archive_member": "./voice.onnx",
        "archive_checksum": sha256(archive).hexdigest(),
        "checksum": sha256(b"expected").hexdigest(),
    }
    assert not _fetch(tmp_path / "voice.onnx", info, archive)
    assert not list(tmp_path.iterdir())
