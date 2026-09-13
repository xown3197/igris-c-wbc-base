#!/usr/bin/env python3
"""Verify the exact allowlisted manufacturer source used by WBC_base."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(repo: Path, lock_path: Path) -> dict[str, object]:
    root = repo.resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != lock["commit"]:
        raise RuntimeError(f"commit mismatch: expected {lock['commit']}, got {commit}")

    verified: dict[str, str] = {}
    for relative, expected in lock["files"].items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root):
            raise RuntimeError(f"locked path escapes repository: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"sha256 mismatch for {relative}: expected {expected}, got {actual}"
            )
        verified[relative] = actual
    return {"commit": commit, "status": "verified", "files": verified}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "upstream.lock.json",
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.repository, args.lock), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

