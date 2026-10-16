"""Signed release packages (``.vpkg`` tar) for adapters: build, Ed25519 sign, verify.

Implements report §2.8 (release). A package holds the adapter weights, μ_new, the approved card, the
open-set calibration (μ(s)/σ(s) tables, conformal scores, α, CUSUM κ/h) and a manifest (semver,
created_at, SHA-256 of every file). ``manifest.sig`` is an Ed25519 signature over ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def generate_keys(d: Path) -> tuple[Path, Path]:
    d.mkdir(parents=True, exist_ok=True)
    sk = Ed25519PrivateKey.generate()
    priv, pub = d / "release_ed25519.pem", d / "release_ed25519.pub.pem"
    priv.write_bytes(sk.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pub.write_bytes(sk.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return priv, pub


def _load_priv(p: Path) -> Ed25519PrivateKey:
    k = serialization.load_pem_private_key(p.read_bytes(), password=None)
    assert isinstance(k, Ed25519PrivateKey)
    return k


def _load_pub(p: Path) -> Ed25519PublicKey:
    k = serialization.load_pem_public_key(p.read_bytes())
    assert isinstance(k, Ed25519PublicKey)
    return k


def build_release(out: Path, files: dict[str, Path], version: str, private_key: Path, extra: dict | None = None) -> Path:
    """Write ``out`` (.vpkg tar) containing ``files`` (arcname -> path), manifest.json and manifest.sig."""
    manifest = {"format": "vimmune-release/1", "version": version, "created_at": datetime.now(timezone.utc).isoformat(),
                "files": {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}, **(extra or {})}
    mbytes = json.dumps(manifest, indent=1, sort_keys=True).encode()
    sig = _load_priv(private_key).sign(mbytes)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tarfile.open(tmp, "w") as tar:
        for n, p in files.items():
            tar.add(p, arcname=n)
        for n, b in (("manifest.json", mbytes), ("manifest.sig", sig)):
            ti = tarfile.TarInfo(n)
            ti.size = len(b)
            tar.addfile(ti, io.BytesIO(b))
    tmp.replace(out)  # atomic: watchers never see a half-written package
    return out


class ReleaseError(Exception):
    pass


def verify_release(path: Path, public_key: Path) -> tuple[dict, dict[str, bytes]]:
    """Check the signature and every file hash. Returns (manifest, {name: bytes})."""
    with tarfile.open(path, "r") as tar:
        blobs = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}  # type: ignore[union-attr]
    if "manifest.json" not in blobs or "manifest.sig" not in blobs:
        raise ReleaseError("missing manifest or signature")
    try:
        _load_pub(public_key).verify(blobs["manifest.sig"], blobs["manifest.json"])
    except Exception as e:
        raise ReleaseError(f"bad signature: {e}") from e
    manifest = json.loads(blobs["manifest.json"])
    for n, h in manifest["files"].items():
        if hashlib.sha256(blobs.get(n, b"")).hexdigest() != h:
            raise ReleaseError(f"hash mismatch for {n}")
    return manifest, {n: b for n, b in blobs.items() if n in manifest["files"]}


def semver_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))
