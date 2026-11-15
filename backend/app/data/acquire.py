"""MIT-BIH acquisition: download from PhysioNet, verify SHA-256, record provenance.

The raw directory is treated as immutable: files are written once, verified against
PhysioNet's published SHA256SUMS.txt and then marked read-only.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from .. import config

REQUIRED_EXT = ("dat", "hea", "atr")
STATE: dict = {"running": False, "stage": "idle", "progress": 0.0, "error": None}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def required_files() -> list[str]:
    return [f"{r}.{e}" for r in config.RECORDS for e in REQUIRED_EXT]


def fetch_checksums(session: requests.Session | None = None) -> dict[str, str]:
    s = session or requests.Session()
    r = s.get(config.DATASET_FILES_URL + "SHA256SUMS.txt", timeout=60)
    r.raise_for_status()
    out = {}
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out


def _set_readonly(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def download(progress: Callable[[str, float], None] | None = None, force: bool = False) -> dict:
    """Download the dataset, verify every required file and write provenance.json."""
    progress = progress or (lambda stage, frac: None)
    raw = config.RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    progress("checksums", 0.0)
    sums = fetch_checksums(session)
    (raw / "SHA256SUMS.txt").write_text(
        "\n".join(f"{v} {k}" for k, v in sorted(sums.items())) + "\n", encoding="utf-8")

    needed = [f for f in required_files()
              if force or not (raw / f).exists() or sha256_file(raw / f) != sums.get(f)]
    method = "already-present"
    if len(needed) > 20:
        method = "zip"
        try:
            progress("download", 0.0)
            buf = io.BytesIO()
            with session.get(config.DATASET_ZIP_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0)) or 77_000_000
                for chunk in r.iter_content(1 << 20):
                    buf.write(chunk)
                    progress("download", min(buf.tell() / total, 1.0))
            progress("extract", 0.0)
            with zipfile.ZipFile(buf) as zf:
                wanted = set(required_files())
                for info in zf.infolist():
                    name = Path(info.filename).name
                    # only top-level record files; ignore auxiliary folders (e.g. x_mitdb)
                    if name in wanted and "x_mitdb" not in info.filename:
                        target = raw / name
                        if target.exists():
                            os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
                        target.write_bytes(zf.read(info))
        except Exception:
            method = "per-file"
    if method in ("per-file",) or (method == "already-present" and needed) or \
            (method == "zip" and any(not (raw / f).exists() for f in required_files())):
        method = "per-file" if method != "zip" else "zip+per-file"
        todo = [f for f in required_files()
                if not (raw / f).exists() or sha256_file(raw / f) != sums.get(f)]
        for i, f in enumerate(todo):
            progress("download", i / max(len(todo), 1))
            r = session.get(config.DATASET_FILES_URL + f, timeout=120)
            r.raise_for_status()
            target = raw / f
            if target.exists():
                os.chmod(target, stat.S_IWRITE | stat.S_IREAD)
            target.write_bytes(r.content)

    progress("verify", 0.0)
    report = verify(sums)
    if not report["ok"]:
        raise RuntimeError(f"Dataset verification failed: {report['problems'][:5]}")
    for f in required_files():
        _set_readonly(raw / f)
    provenance = {
        "dataset": config.DATASET_NAME,
        "version": config.DATASET_VERSION,
        "source_url": config.DATASET_URL,
        "files_url": config.DATASET_FILES_URL,
        "doi": config.DATASET_DOI,
        "license": config.DATASET_LICENSE,
        "citations": config.DATASET_CITATIONS,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "download_method": method,
        "n_files_verified": report["n_verified"],
        "checksum_source": config.DATASET_FILES_URL + "SHA256SUMS.txt",
        "sha256": {f: sums[f] for f in required_files()},
    }
    (raw / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    progress("done", 1.0)
    return provenance


def verify(sums: dict[str, str] | None = None) -> dict:
    """Check that every required file exists and matches its published SHA-256."""
    raw = config.RAW_DIR
    if sums is None:
        p = raw / "SHA256SUMS.txt"
        sums = {}
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2:
                    sums[parts[1]] = parts[0]
    problems, n_ok = [], 0
    for f in required_files():
        path = raw / f
        if not path.exists():
            problems.append(f"missing: {f}")
        elif f not in sums:
            problems.append(f"no published checksum: {f}")
        elif sha256_file(path) != sums[f]:
            problems.append(f"checksum mismatch: {f}")
        else:
            n_ok += 1
    return {"ok": not problems, "n_verified": n_ok, "n_required": len(required_files()),
            "problems": problems}


def is_present() -> bool:
    return all((config.RAW_DIR / f).exists() for f in required_files())


def provenance() -> dict | None:
    p = config.RAW_DIR / "provenance.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
