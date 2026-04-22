"""Runtime configuration for the Modal RF-DETR SAHI inference service.

All values come from environment variables so that deployment-specific
settings (Modal environment, Volume name, Label Studio base URL, label
mapping) stay out of source control. Defaults favour a local-first
self-hosted deployment with a single Modal L4 GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_str_with_default(name: str, default: str) -> str:
    value = _env_str(name, default)
    assert value is not None
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class ModalSettings:
    modal_app_name: str
    modal_environment: str | None
    volume_name: str
    volume_mount_path: str
    gpu_type: str | None
    request_timeout_seconds: int
    model_version: str
    task_image_field: str
    label_studio_base_url: str | None
    label_studio_auth_header: str | None
    label_studio_mapping_json: str | None
    label_studio_from_name: str
    label_studio_to_name: str
    default_confidence_threshold: float
    default_overlap_fraction: float
    default_wbf_iou_threshold: float
    image_request_timeout_seconds: float


def get_modal_settings() -> ModalSettings:
    label_studio_auth_header = _env_str("LABEL_STUDIO_AUTHORIZATION")
    label_studio_api_key = _env_str("LABEL_STUDIO_API_KEY")
    label_studio_api_key_type = _env_str_with_default(
        "LABEL_STUDIO_API_KEY_TYPE", "Token"
    )
    if label_studio_auth_header is None and label_studio_api_key is not None:
        label_studio_auth_header = f"{label_studio_api_key_type} {label_studio_api_key}"

    return ModalSettings(
        modal_app_name=_env_str_with_default("MODAL_APP_NAME", "dermoscopy-rfdetr-sahi-inference"),
        modal_environment=_env_str("MODAL_ENVIRONMENT"),
        volume_name=_env_str_with_default("MODAL_VOLUME_NAME", "dermoscopy-rfdetr-checkpoints"),
        volume_mount_path=_env_str_with_default("VOLUME_MOUNT_PATH", "/models"),
        gpu_type=_env_str("MODAL_GPU", "L4"),
        request_timeout_seconds=_env_int("MODAL_REQUEST_TIMEOUT", 900),
        model_version=_env_str_with_default("MODEL_VERSION", "rfdetr-sahi-v1"),
        task_image_field=_env_str_with_default("TASK_IMAGE_FIELD", "image"),
        label_studio_base_url=_env_str("LABEL_STUDIO_BASE_URL"),
        label_studio_auth_header=label_studio_auth_header,
        label_studio_mapping_json=_env_str("LABEL_STUDIO_MAPPING_JSON"),
        label_studio_from_name=_env_str_with_default("LABEL_STUDIO_FROM_NAME", "label"),
        label_studio_to_name=_env_str_with_default("LABEL_STUDIO_TO_NAME", "image"),
        default_confidence_threshold=_env_float("DEFAULT_CONFIDENCE_THRESHOLD", 0.15),
        default_overlap_fraction=_env_float("DEFAULT_OVERLAP_FRACTION", 0.25),
        default_wbf_iou_threshold=_env_float("DEFAULT_WBF_IOU_THRESHOLD", 0.55),
        image_request_timeout_seconds=_env_float("IMAGE_REQUEST_TIMEOUT", 30.0),
    )


def resolve_gpu(gpu_type: str | None) -> str | None:
    """Normalise the GPU string accepted by Modal's ``@app.cls(gpu=...)``.

    Returns ``None`` when the caller opted out of GPU scheduling (empty
    string, ``"none"`` or ``"cpu"``), otherwise forwards the trimmed
    value which Modal accepts (e.g. ``"L4"``, ``"A10G"``, ``"H100"``).
    """

    if gpu_type is None:
        return None
    normalised = gpu_type.strip()
    if normalised == "" or normalised.lower() in ("none", "cpu"):
        return None
    return normalised
