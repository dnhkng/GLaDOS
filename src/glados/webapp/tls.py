"""Self-signed HTTPS for the console, so phones on the LAN may use their microphone."""

from __future__ import annotations

import ipaddress
from pathlib import Path
import shutil
import ssl
import subprocess

DEFAULT_DIR = Path("data/webapp_tls")


def _subject_alt_names(hosts: frozenset[str]) -> str:
    names = []
    for host in sorted(hosts):
        try:
            ipaddress.ip_address(host)
            names.append("IP:" + host)
        except ValueError:
            names.append("DNS:" + host)
    return ",".join(names)


def ensure_certificate(hosts: frozenset[str], directory: Path = DEFAULT_DIR) -> tuple[Path, Path]:
    """Return a certificate and key covering ``hosts``, generating a new pair when the hosts change."""
    cert, key, names = directory / "cert.pem", directory / "key.pem", directory / "hosts.txt"
    san = _subject_alt_names(hosts)
    if cert.exists() and key.exists() and names.exists() and names.read_text() == san:
        return cert, key
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl is required to generate the console's HTTPS certificate")
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [openssl, "req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
         "-days", "825", "-subj", "/CN=GLaDOS console", "-addext", "subjectAltName=" + san,
         "-keyout", str(key), "-out", str(cert)],
        check=True, capture_output=True, timeout=30,
    )
    key.chmod(0o600)
    names.write_text(san)
    return cert, key


def server_context(cert: Path, key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    return context
