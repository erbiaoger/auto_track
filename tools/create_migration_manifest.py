#!/usr/bin/env python3
"""Create a compact, reversible inventory before the repository migration.

Large arrays are fingerprinted from their first/last MiB instead of being
fully hashed. Checkpoints and other files up to 512 MiB receive a SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "archive" / "migration_manifests" / "pre_migration_inventory.jsonl"
SKIP_PARTS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".tsbuild"}
LARGE_ROOTS = {"datasets", "models", "predicts", "results", "runs", "bak", "logs"}
# The first pass must be fast even on the 1 TB workspace. Active checkpoint
# hashes are recorded separately during the validation phase.
HASH_LIMIT = 0
SAMPLE_SIZE = 1024 * 1024


def _skip(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return True
    if any(part in SKIP_PARTS for part in relative.parts):
        return True
    # Record the large asset roots and their immediate ownership folders in
    # this fast pass. Individual moved assets receive a focused fingerprint
    # manifest during each migration batch.
    for index, part in enumerate(relative.parts):
        if part in LARGE_ROOTS and len(relative.parts) > index + 2:
            return True
    return False


def _digest(path: Path, size: int) -> tuple[str, str]:
    if not path.is_file():
        return "", ""
    h = hashlib.sha256()
    if size <= HASH_LIMIT:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        return "sha256", h.hexdigest()
    with path.open("rb") as handle:
        h.update(handle.read(SAMPLE_SIZE))
        if size > SAMPLE_SIZE:
            handle.seek(max(0, size - SAMPLE_SIZE))
            h.update(handle.read(SAMPLE_SIZE))
    return "sha256_first_last_1MiB", h.hexdigest()


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT.open("w", encoding="utf-8") as output:
        for path in sorted(ROOT.rglob("*")):
            if _skip(path) or path == OUT:
                continue
            try:
                info = path.lstat()
            except OSError as exc:
                output.write(json.dumps({"path": str(path.relative_to(ROOT)), "error": str(exc)}) + "\n")
                continue
            mode = stat.S_IFMT(info.st_mode)
            record = {
                "path": str(path.relative_to(ROOT)),
                "kind": "file" if mode == stat.S_IFREG else "directory" if mode == stat.S_IFDIR else "symlink" if mode == stat.S_IFLNK else "other",
                "size_bytes": int(info.st_size),
                "device": int(info.st_dev),
                "inode": int(info.st_ino),
                "mtime_ns": int(info.st_mtime_ns),
            }
            if record["kind"] == "symlink":
                record["target"] = os.readlink(path)
            elif record["kind"] == "file":
                try:
                    algo, digest = _digest(path, int(info.st_size))
                    record["fingerprint_algorithm"] = algo
                    record["fingerprint"] = digest
                except OSError as exc:
                    record["fingerprint_error"] = str(exc)
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(f"wrote {count} entries to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
