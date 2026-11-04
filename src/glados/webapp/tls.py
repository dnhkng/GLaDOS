"""HTTPS for the console, so phones on the LAN may use their microphone.

A small local certificate authority signs the console certificate. Install ``ca.pem`` on a
phone once (the console offers it at /glados-ca.crt) and every later certificate, regenerated
when the PC changes network, is trusted without warnings. The authority is name-constrained to
private addresses and local names, so even a leaked key cannot impersonate a public website.
"""

from __future__ import annotations

import ipaddress
from pathlib import Path
import shutil
import ssl
import subprocess
import tempfile

DEFAULT_DIR = Path("data/webapp_tls")
_PRIVATE_V4 = ("10.0.0.0/255.0.0.0", "172.16.0.0/255.240.0.0", "192.168.0.0/255.255.0.0", "127.0.0.0/255.0.0.0",
               "169.254.0.0/255.255.0.0")
_NAME_CONSTRAINTS = "critical," + ",".join(
    [f"permitted;IP:{net}" for net in _PRIVATE_V4]
    + ["permitted;IP:::1/ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", "permitted;IP:fc00::/fe00::",
       "permitted;DNS:localhost", "permitted;DNS:.local", "permitted;DNS:.lan", "permitted;DNS:.home.arpa"])


def certifiable(host: str) -> bool:
    """Whether the local authority may sign ``host`` (a private address or a local name)."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host == "localhost" or host.endswith((".local", ".lan", ".home.arpa"))
    return ip.is_loopback or ip.is_private and not ip.is_global


def _subject_alt_names(hosts: frozenset[str]) -> str:
    names = []
    for host in sorted(hosts):
        try:
            ipaddress.ip_address(host)
            names.append("IP:" + host)
        except ValueError:
            names.append("DNS:" + host)
    return ",".join(names)


def _openssl(*args: str) -> None:
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl is required to generate the console's HTTPS certificate")
    subprocess.run([openssl, *args], check=True, capture_output=True, timeout=30)


def ensure_authority(directory: Path = DEFAULT_DIR) -> tuple[Path, Path]:
    """Return the local authority's certificate and key, creating them on first use."""
    ca, ca_key = directory / "ca.pem", directory / "ca-key.pem"
    if ca.exists() and ca_key.exists():
        return ca, ca_key
    directory.mkdir(parents=True, exist_ok=True)
    _openssl("req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes", "-days", "3650",
             "-subj", "/CN=GLaDOS local console authority",
             "-addext", "basicConstraints=critical,CA:TRUE,pathlen:0",
             "-addext", "keyUsage=critical,keyCertSign,cRLSign",
             "-addext", "nameConstraints=" + _NAME_CONSTRAINTS,
             "-keyout", str(ca_key), "-out", str(ca))
    ca_key.chmod(0o600)
    return ca, ca_key


def ensure_certificate(hosts: frozenset[str], directory: Path = DEFAULT_DIR) -> tuple[Path, Path]:
    """Return a certificate and key covering ``hosts``, issuing a new pair when the hosts change.

    Hosts the authority may not sign (public addresses) are left out of the certificate.
    """
    ca, ca_key = ensure_authority(directory)
    cert, key, names = directory / "cert.pem", directory / "key.pem", directory / "hosts.txt"
    san = _subject_alt_names(frozenset(host for host in hosts if certifiable(host)))
    stamp = "ca-signed " + san
    if cert.exists() and key.exists() and names.exists() and names.read_text() == stamp:
        return cert, key
    with tempfile.TemporaryDirectory() as scratch:
        request, extensions = Path(scratch) / "console.csr", Path(scratch) / "console.ext"
        extensions.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\n"
                              f"extendedKeyUsage=serverAuth\nsubjectAltName={san}\n")
        _openssl("req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-nodes",
                 "-subj", "/CN=GLaDOS console", "-keyout", str(key), "-out", str(request))
        # 825 days is the longest validity iOS accepts for a server certificate.
        _openssl("x509", "-req", "-in", str(request), "-CA", str(ca), "-CAkey", str(ca_key), "-CAcreateserial",
                 "-days", "825", "-sha256", "-extfile", str(extensions), "-out", str(cert))
    key.chmod(0o600)
    names.write_text(stamp)
    return cert, key


def server_context(cert: Path, key: Path) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert, key)
    return context
