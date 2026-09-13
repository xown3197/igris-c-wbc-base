#!/usr/bin/env python3
"""Run a hash-locked manufacturer launcher with WBC_base task registration.

The upstream file is executed in memory.  Its bytes on disk are never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


IMPORT_NEEDLE = "import robros_lab_public.tasks  # noqa: F401"
WBC_IMPORT = "import wbc_base.tasks  # noqa: F401"
COMPAT_IMPORT = "from wbc_base.compat import install_action_term_cfg_alias"
COMPAT_CALL = "install_action_term_cfg_alias()"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inject_wbc_tasks(source: str) -> str:
    """Register WBC_base immediately after the manufacturer's task package."""

    if source.count(IMPORT_NEEDLE) != 1:
        raise RuntimeError(
            f"expected exactly one task import marker, got {source.count(IMPORT_NEEDLE)}"
        )
    if WBC_IMPORT in source:
        raise RuntimeError("upstream launcher already imports wbc_base.tasks")
    replacement = "\n".join(
        (COMPAT_IMPORT, COMPAT_CALL, IMPORT_NEEDLE, WBC_IMPORT)
    )
    return source.replace(IMPORT_NEEDLE, replacement)


def resolve_verified_entrypoint(
    upstream_root: Path, mode: str, lock_path: Path
) -> Path:
    root = upstream_root.resolve(strict=True)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entry_rel = f"scripts/rsl_rl/base/{mode}.py"
    expected = lock["files"].get(entry_rel)
    if expected is None:
        raise RuntimeError(f"entrypoint is not allowlisted: {entry_rel}")
    entry = (root / entry_rel).resolve(strict=True)
    if not entry.is_relative_to(root):
        raise RuntimeError(f"entrypoint escapes upstream repository: {entry}")
    actual = sha256(entry)
    if actual != expected:
        raise RuntimeError(
            f"entrypoint sha256 mismatch: expected {expected}, got {actual}"
        )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("mode", choices=("train", "play"))
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--isaaclab-root",
        type=Path,
        help="IsaacLab checkout used to resolve the upstream cli_args module",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "upstream.lock.json",
    )
    args, passthrough = parser.parse_known_args()

    entry = resolve_verified_entrypoint(args.upstream_root, args.mode, args.lock)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    upstream_source = args.upstream_root.resolve() / "source" / "robros_lab_public"
    extra_paths = [os.fspath(project_root / "src"), os.fspath(upstream_source)]
    if args.isaaclab_root is not None:
        isaaclab_rsl_scripts = (
            args.isaaclab_root.resolve(strict=True)
            / "scripts"
            / "reinforcement_learning"
            / "rsl_rl"
        )
        if not isaaclab_rsl_scripts.is_dir():
            raise RuntimeError(
                f"IsaacLab RSL-RL scripts not found: {isaaclab_rsl_scripts}"
            )
        extra_paths.append(os.fspath(isaaclab_rsl_scripts))
    sys.path[:0] = extra_paths
    sys.argv = [os.fspath(entry), *passthrough]
    os.chdir(output_root)

    source = inject_wbc_tasks(entry.read_text(encoding="utf-8"))
    globals_dict = {
        "__name__": "__main__",
        "__file__": os.fspath(entry),
        "__package__": None,
        "__cached__": None,
    }
    exec(compile(source, os.fspath(entry), "exec"), globals_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
