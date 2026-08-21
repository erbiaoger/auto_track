"""Validate method/shared-data catalogs without copying large arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _path(root: Path, catalog: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (catalog.parent / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    catalogs = [root / "shared_data/catalog.yaml"]
    catalogs.extend(root.glob("methods/*/data/catalog.yaml"))
    catalogs.extend(root.glob("methods/*/checkpoints/catalog.yaml"))
    catalogs.extend(root.glob("methods/*/results/catalog.yaml"))
    errors: list[str] = []
    checked = 0
    for catalog in sorted(catalogs):
        payload = yaml.safe_load(catalog.read_text(encoding="utf-8")) or {}
        for entry in payload.get("entries", []):
            value = entry.get("path")
            if not value or entry.get("status") in {"external", "broken"}:
                continue
            target = _path(root, catalog, str(value))
            checked += 1
            if not target.exists():
                errors.append(f"{catalog.relative_to(root)}: missing {value}")
        if catalog.name == "catalog.yaml" and catalog.parent.name == "data" and catalog.parent.parent.name == "shared_data":
            raw = root / "shared_data/day11/source/raw_DAY11.npy"
            if raw.exists():
                array = np.load(raw, mmap_mode="r")
                if tuple(array.shape) != (3_600_000, 50) or array.dtype != np.float32:
                    errors.append(f"{raw.relative_to(root)}: expected float32 [3600000,50], got {array.dtype} {array.shape}")
    registry = root / "vehicle_replay_web/backend/methods.yaml"
    methods = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    for method in methods.get("methods", []):
        for key in ("checkpoint", "config"):
            value = method.get(key)
            if value and not (root / value).exists():
                errors.append(f"registry {method.get('id')}: missing {key} {value}")
    report = {"catalogs": len(catalogs), "checked_entries": checked, "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
