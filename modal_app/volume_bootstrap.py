"""Synchronise deployment-ready RF-DETR checkpoints into the Modal Volume.

The script reads the deterministic registry in
``modal_app/model_registry.py`` and uploads every supported checkpoint
from ``models_deployment/`` into stable paths inside a named Modal
Volume. Each run replaces the currently served checkpoint files with the
local copies, so rerunning the script updates the deployment in-place.

Run from the repository root::

    python -m modal_app.volume_bootstrap \
        --volume-name dermoscopy-rfdetr-checkpoints \
        [--environment prod]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import get_modal_settings
from .model_registry import (
    LOCAL_MODELS_ROOT,
    MODEL_REGISTRY,
    ModelEntry,
    remote_checkpoint_volume_path,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_modal_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume-name",
        default=settings.volume_name,
        help="Name of the Modal Volume (default: env MODAL_VOLUME_NAME).",
    )
    parser.add_argument(
        "--environment",
        default=settings.modal_environment,
        help="Optional Modal environment name (default: env MODAL_ENVIRONMENT).",
    )
    parser.add_argument(
        "--models-root",
        default=str(LOCAL_MODELS_ROOT),
        help=(
            "Local directory holding deployment-ready checkpoints "
            "(default: models_deployment). Each checkpoint is resolved as "
            "<models-root>/<remote_filename>."
        ),
    )
    return parser.parse_args(argv)


def _resolve_local_checkpoint(entry: ModelEntry, models_root: str) -> Path:
    return Path(models_root) / entry.remote_checkpoint_filename


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    missing: list[str] = []
    uploads: list[tuple[Path, str]] = []
    for entry in MODEL_REGISTRY:
        local = _resolve_local_checkpoint(entry, args.models_root)
        if not local.is_file():
            missing.append(f"{entry.class_key}: {local}")
            continue
        uploads.append((local, remote_checkpoint_volume_path(entry)))

    if missing:
        joined = "\n  ".join(missing)
        print(
            f"[bootstrap] ERROR: local checkpoints are missing:\n  {joined}",
            file=sys.stderr,
        )
        return 2

    import modal

    volume = modal.Volume.from_name(
        args.volume_name,
        environment_name=args.environment,
        create_if_missing=True,
    )

    environment_label = args.environment or "default"
    print(
        f"[bootstrap] volume={args.volume_name} environment={environment_label} "
        f"replace_existing=True uploads={len(uploads)} models_root={args.models_root}"
    )

    with volume.batch_upload(force=True) as batch:
        for local, remote in uploads:
            batch.put_file(local, remote)
            print(f"[bootstrap] queued {local} -> volume:{remote}")

    print("[bootstrap] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
