"""Registry of RF-DETR checkpoints served by the Modal inference service.

The deployment-ready checkpoints now live under ``models_deployment/``
with stable filenames that are uploaded to the Modal Volume as-is. The
registry still provides one deterministic entry per supported class,
aligned with the category ids declared in
``dermoscopy_dataset/result.json``. Classes that do not have a trained
checkpoint are intentionally absent so they never leak into
predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LOCAL_MODELS_ROOT = Path("models_deployment")

REMOTE_CHECKPOINT_SUBDIR = "checkpoints"


@dataclass(frozen=True)
class ModelEntry:
    class_key: str
    result_json_name: str
    category_id: int
    variant: str
    local_checkpoint_path: Path
    remote_checkpoint_filename: str
    training_patch_size: int


MODEL_REGISTRY: tuple[ModelEntry, ...] = (
    ModelEntry(
        class_key="blue_gray_globules",
        result_json_name="Blue-gray globules",
        category_id=1,
        variant="nano",
        local_checkpoint_path=LOCAL_MODELS_ROOT / "blue_gray_globules.pth",
        remote_checkpoint_filename="blue_gray_globules.pth",
        training_patch_size=128,
    ),
    ModelEntry(
        class_key="may_globules",
        result_json_name="MAY globules",
        category_id=4,
        variant="nano",
        local_checkpoint_path=LOCAL_MODELS_ROOT / "may_globules.pth",
        remote_checkpoint_filename="may_globules.pth",
        training_patch_size=256,
    ),
    ModelEntry(
        class_key="milia_like_cyst",
        result_json_name="Milia-like-cyst",
        category_id=5,
        variant="nano",
        local_checkpoint_path=LOCAL_MODELS_ROOT / "milia_like_cyst.pth",
        remote_checkpoint_filename="milia_like_cyst.pth",
        training_patch_size=256,
    ),
    ModelEntry(
        class_key="rosettes",
        result_json_name="Rosettes",
        category_id=6,
        variant="nano",
        local_checkpoint_path=LOCAL_MODELS_ROOT / "rosettes.pth",
        remote_checkpoint_filename="rosettes.pth",
        training_patch_size=256,
    ),
    ModelEntry(
        class_key="yellow_globlues_ulcer",
        result_json_name="Yellow globlues (ulcer)",
        category_id=9,
        variant="medium",
        # Keep the historic class_key used by the API/config surface, but
        # map it to the corrected deployment filename.
        local_checkpoint_path=LOCAL_MODELS_ROOT / "yellow_globules_ulcer.pth",
        remote_checkpoint_filename="yellow_globules_ulcer.pth",
        training_patch_size=512,
    ),
)


SUPPORTED_CLASS_KEYS: frozenset[str] = frozenset(entry.class_key for entry in MODEL_REGISTRY)


def remote_checkpoint_mount_path(volume_mount_path: str | Path, entry: ModelEntry) -> Path:
    """Absolute in-container path where a checkpoint is mounted."""

    return Path(volume_mount_path) / REMOTE_CHECKPOINT_SUBDIR / entry.remote_checkpoint_filename


def remote_checkpoint_volume_path(entry: ModelEntry) -> str:
    """Volume-internal remote path used when uploading a checkpoint.

    Modal volumes are addressed with leading-slash paths relative to the
    volume root, so uploads and runtime mounts must share the same
    sub-tree (``/<REMOTE_CHECKPOINT_SUBDIR>/<filename>``).
    """

    return f"/{REMOTE_CHECKPOINT_SUBDIR}/{entry.remote_checkpoint_filename}"


def default_label_for_category_id() -> dict[int, str]:
    """Default ``category_id -> Label Studio label name`` mapping.

    Falls back to ``result.json`` class names so a Label Studio config
    that reuses those exact strings works without any override.
    """

    return {entry.category_id: entry.result_json_name for entry in MODEL_REGISTRY}


def get_entry(class_key: str) -> ModelEntry:
    for entry in MODEL_REGISTRY:
        if entry.class_key == class_key:
            return entry
    raise KeyError(f"Unknown class_key: {class_key!r}")
