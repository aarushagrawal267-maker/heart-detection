"""Command-line helpers used by setup.ps1 and for reproducibility.

    python -m app.cli status      # dataset / model status
    python -m app.cli download    # download + verify MIT-BIH, build beat set
    python -m app.cli verify      # re-verify SHA-256 checksums
    python -m app.cli build       # rebuild the processed beat set
    python -m app.cli init-db     # create the SQLite schema
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .data import acquire
from .db import models as db
from .segmentation import beats


def _progress(stage: str, frac: float) -> None:
    sys.stdout.write(f"\r  {stage:<12} {frac * 100:5.1f}%")
    sys.stdout.flush()
    if stage == "done":
        sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.cli")
    p.add_argument("command", choices=["status", "download", "verify", "build", "init-db"])
    args = p.parse_args(argv)

    if args.command == "init-db":
        db.engine()
        from .training.registry import import_all
        n = import_all()
        print(f"SQLite database ready: {config.DB_PATH} ({n} bundled experiments imported)")
    elif args.command == "download":
        print(f"Downloading {config.DATASET_NAME} {config.DATASET_VERSION} from PhysioNet ...")
        prov = acquire.download(progress=_progress)
        print(f"\nVerified {prov['n_files_verified']} files against SHA256SUMS.txt -> {config.RAW_DIR}")
        if not beats.is_built():
            print("Building processed beat set ...")
            man = beats.build(progress=_progress)
            print(f"{man['n_segments']} beat segments written to {config.processed_dir()}")
    elif args.command == "verify":
        rep = acquire.verify()
        print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=2))
        for prob in rep["problems"][:20]:
            print("  ", prob)
        return 0 if rep["ok"] else 1
    elif args.command == "build":
        man = beats.build(progress=_progress)
        print(f"{man['n_segments']} beat segments written to {config.processed_dir()}")
    elif args.command == "status":
        out = {"dataset_present": acquire.is_present(), "processed": beats.is_built(),
               "raw_dir": str(config.RAW_DIR), "db": str(config.DB_PATH)}
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
