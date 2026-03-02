"""Single-file model packaging/unpackaging (.sfog archive)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from sfogliatella.core.utils import get_version_info

logger = logging.getLogger(__name__)

PACK_EXTENSION = ".sfog"
PACK_MANIFEST  = "manifest.json"


def pack_model(
    model_dir: Path,
    run_id: str,
    output_path: Optional[Path] = None,
) -> Path:
    """Package model_dir/run_id into a single .sfog archive.

    The archive is a .tar.gz containing:
      - manifest.json (version info, run_id, config)
      - weights/     (all checkpoint files)
      - config.json
    """
    model_dir = Path(model_dir)
    run_dir = model_dir / run_id

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if output_path is None:
        output_path = model_dir / f"{run_id}{PACK_EXTENSION}"
    output_path = Path(output_path)

    # Build manifest
    config_file = run_dir / "config.json"
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    manifest = {
        "sfogliatella_version": get_version_info().get("sfogliatella", "unknown"),
        "run_id": run_id,
        "config": config,
        "versions": get_version_info(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Write manifest
        (tmp / PACK_MANIFEST).write_text(json.dumps(manifest, indent=2))
        # Copy run_dir contents
        shutil.copytree(run_dir, tmp / "run", dirs_exist_ok=False)

        # Create tar.gz
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(tmp, arcname="sfog")

    logger.info("Model packed: %s", output_path)
    return output_path


def unpack_model(
    pack_path: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Unpack a .sfog archive into output_dir.

    Returns the run directory path.
    """
    pack_path = Path(pack_path)
    if not pack_path.exists():
        raise FileNotFoundError(f"Pack file not found: {pack_path}")

    if output_dir is None:
        output_dir = pack_path.parent / pack_path.stem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(pack_path, "r:gz") as tar:
        tar.extractall(output_dir)

    # Find manifest
    manifest_path = output_dir / "sfog" / PACK_MANIFEST
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        run_id = manifest.get("run_id", "unknown")
        logger.info("Unpacked %s (run_id=%s)", pack_path, run_id)
    else:
        logger.warning("No manifest found in archive")

    run_dir = output_dir / "sfog" / "run"
    logger.info("Model unpacked to: %s", run_dir)
    return run_dir


def load_manifest(pack_path: Path) -> Dict[str, Any]:
    """Read manifest from a .sfog archive without fully unpacking."""
    with tarfile.open(pack_path, "r:gz") as tar:
        try:
            member = tar.getmember(f"sfog/{PACK_MANIFEST}")
            f = tar.extractfile(member)
            return json.load(f)
        except KeyError:
            raise FileNotFoundError(f"No manifest in archive: {pack_path}")
