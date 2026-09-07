"""Run manifests (MASTER_PROMPT section 15).

Every run writes ``results/raw/<version>/manifest.json`` carrying the seed, the
full config, the git commit, package versions, dataset checksums and metrics.
Figures are generated from manifests only, so anything a figure shows has to be
recorded here first.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import paths

TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "timm",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "pillow",
    "opencv-python",
)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=paths.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            info["gpu_total_mem_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            )
    except Exception:  # noqa: BLE001
        pass
    return info


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def dataset_checksums() -> dict[str, str]:
    """Checksum of the verified index rather than of ~2 GB of pixels.

    The index is derived from every file on disk, so a changed dataset changes
    this digest.
    """
    out: dict[str, str] = {}
    idx = paths.DATA_DIR / "index.parquet"
    if idx.exists():
        out["index.parquet"] = hashlib.sha256(idx.read_bytes()).hexdigest()
    return out


def write_manifest(
    version: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    seed: int,
    extra: dict[str, Any] | None = None,
    filename: str = "manifest.json",
) -> Path:
    out_dir = paths.RAW_RESULTS / version
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": version,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "config": config,
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "packages": package_versions(),
        "hardware": hardware(),
        "dataset_checksums": dataset_checksums(),
        "metrics": metrics,
    }
    if extra:
        manifest.update(extra)
    path = out_dir / filename
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def load_manifest(version: str, filename: str = "manifest.json") -> dict[str, Any]:
    path = paths.RAW_RESULTS / version / filename
    return json.loads(path.read_text(encoding="utf-8"))
