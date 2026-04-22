"""Label Studio compatibility helpers for the RF-DETR SAHI inference service.

Three concerns live here:

* Pydantic schemas for the Label Studio ML backend request / response
  contract plus validation of the per-class SAHI overrides,
* resolving the image referenced by a Label Studio task payload (HTTP,
  ``file://`` URI or a local filesystem path; authentication is
  intentionally not performed because the Label Studio instance is
  configured without auth),
* converting aggregated predictions into Label Studio pre-annotation
  results with coordinates expressed as percentages of the original
  image size.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PerClassParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    patch_size: int | None = None
    confidence_threshold: float | None = None

    @field_validator("patch_size")
    @classmethod
    def _check_patch_size(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("patch_size must be a positive integer")
        return value

    @field_validator("confidence_threshold")
    @classmethod
    def _check_confidence(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        return value


class PredictionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overlap_fraction: float | None = None
    wbf_iou_threshold: float | None = None
    per_class: dict[str, PerClassParams] = Field(default_factory=dict)

    @field_validator("overlap_fraction")
    @classmethod
    def _check_overlap(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value < 1.0:
            raise ValueError("overlap_fraction must lie in [0, 1)")
        return value

    @field_validator("wbf_iou_threshold")
    @classmethod
    def _check_wbf_iou(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 < value <= 1.0:
            raise ValueError("wbf_iou_threshold must lie in (0, 1]")
        return value


class PredictRequestParams(PredictionParams):
    """``params`` payload accepted by ``POST /predict``.

    Label Studio injects transport-level fields such as ``context``,
    ``login`` and ``password`` alongside model-specific params. The
    backend ignores those fields and validates only the SAHI overrides
    relevant to this service.
    """

    model_config = ConfigDict(extra="ignore")

    context: dict[str, Any] | None = None
    login: str | None = None
    password: str | None = None


class LabelStudioTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int | str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    tasks: list[LabelStudioTask]
    params: PredictRequestParams | None = None


class SetupRequest(BaseModel):
    """Payload Label Studio sends to ``POST /setup``.

    Label Studio forwards the *Extra Params* JSON from the Connect Model
    dialog as ``extra_params``. Depending on the Label Studio version,
    the field may arrive either as an already-parsed JSON object, a raw
    JSON string copied from the textarea, an empty string, or be absent
    entirely. Other meta-fields (``project``, ``schema``, ``hostname``,
    ...) are tolerated via ``extra="allow"`` so the backend does not
    need to model them explicitly.
    """

    model_config = ConfigDict(extra="allow")

    extra_params: dict[str, Any] | str | None = None
    hostname: str | None = None


def first_non_none(*values):
    """Return the first argument that is not ``None``, else ``None``.

    Used by ``/predict`` to walk the precedence chain
    ``per-call request body > /setup extra_params > env / registry``.
    """

    for value in values:
        if value is not None:
            return value
    return None


def parse_setup_extra_params(extra_params: dict[str, Any] | str | None) -> dict[str, Any]:
    """Normalise Label Studio ``extra_params`` to a JSON object.

    The Connect Model dialog stores extra params in a textarea, so some
    Label Studio versions send the field as a raw string while others
    send a parsed object. Empty input is treated as "no runtime
    defaults".
    """

    if extra_params is None:
        return {}
    if isinstance(extra_params, dict):
        return extra_params
    if isinstance(extra_params, str):
        stripped = extra_params.strip()
        if stripped == "":
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra_params is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("extra_params JSON must decode to an object.")
        return parsed
    raise TypeError(
        "extra_params must be either a JSON object, a JSON string, or an empty value."
    )


def derive_label_studio_base_url(
    configured_base_url: str | None,
    setup_hostname: str | None,
) -> str | None:
    """Resolve the Label Studio base URL used for relative image paths.

    Preference order:
    1. explicit environment configuration,
    2. hostname delivered by Label Studio during ``POST /setup``.
    """

    for candidate in (configured_base_url, setup_hostname):
        normalised = _normalise_base_url(candidate)
        if normalised is not None:
            return normalised
    return None


def _normalise_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped == "":
        return None

    parsed = urlparse(stripped)
    if parsed.scheme in ("http", "https"):
        return stripped.rstrip("/")

    # Some Label Studio payloads may send just the host[:port].
    if parsed.scheme == "" and not stripped.startswith("/") and " " not in stripped:
        assumed_scheme = "http" if stripped.startswith(("localhost", "127.0.0.1")) else "https"
        return f"{assumed_scheme}://{stripped}".rstrip("/")

    return None


class LabelStudioMapping(BaseModel):
    from_name: str
    to_name: str
    labels: dict[int, str]


def load_label_studio_mapping(
    mapping_json: str | None,
    default_from_name: str,
    default_to_name: str,
    default_labels: dict[int, str],
) -> LabelStudioMapping:
    """Build the Label Studio mapping from environment-provided JSON.

    When ``mapping_json`` is ``None`` the caller's defaults are used
    untouched. Otherwise the JSON may define ``from_name``, ``to_name``
    and/or a ``labels`` object keyed by ``category_id``. Missing keys
    fall back to the defaults.
    """

    from_name = default_from_name
    to_name = default_to_name
    labels = dict(default_labels)

    if mapping_json is not None:
        try:
            raw = json.loads(mapping_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LABEL_STUDIO_MAPPING_JSON is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("LABEL_STUDIO_MAPPING_JSON must decode to a JSON object.")
        if "from_name" in raw:
            from_name = str(raw["from_name"])
        if "to_name" in raw:
            to_name = str(raw["to_name"])
        labels_raw = raw.get("labels", {})
        if not isinstance(labels_raw, dict):
            raise ValueError("LABEL_STUDIO_MAPPING_JSON 'labels' must be an object.")
        for key, value in labels_raw.items():
            labels[int(key)] = str(value)

    return LabelStudioMapping(from_name=from_name, to_name=to_name, labels=labels)


def resolve_image(
    task_data: dict[str, Any],
    task_image_field: str,
    base_url: str | None,
    timeout: float,
    auth_header: str | None = None,
    basic_auth: tuple[str, str] | None = None,
) -> Image.Image:
    """Load the image referenced by a Label Studio task payload."""

    if task_image_field in task_data:
        raw_value = task_data[task_image_field]
    else:
        fallback_candidates = {
            key: value
            for key, value in task_data.items()
            if isinstance(value, str) and _looks_like_image_reference(value)
        }
        if len(fallback_candidates) == 1:
            fallback_key, raw_value = next(iter(fallback_candidates.items()))
            print(
                "[label-studio-adapter] task image field fallback engaged: "
                f"configured={task_image_field!r} resolved={fallback_key!r}"
            )
        else:
            raise KeyError(
                f"Task payload is missing the image field {task_image_field!r}; "
                f"available keys: {sorted(task_data.keys())}"
            )
    if not isinstance(raw_value, str) or not raw_value:
        raise TypeError(
            f"Task field {task_image_field!r} must be a non-empty string, "
            f"got {type(raw_value).__name__}"
        )

    parsed = urlparse(raw_value)
    if parsed.scheme in ("http", "https"):
        url = raw_value
    elif parsed.scheme == "file":
        local_path = Path(parsed.path)
        with Image.open(local_path) as raw:
            return raw.convert("RGB")
    elif parsed.scheme == "":
        local_candidate = Path(raw_value)
        if local_candidate.is_file():
            with Image.open(local_candidate) as raw:
                return raw.convert("RGB")
        if not base_url:
            raise ValueError(
                f"Image reference {raw_value!r} is relative and LABEL_STUDIO_BASE_URL is not configured."
            )
        if raw_value.startswith("/"):
            url = base_url.rstrip("/") + raw_value
        else:
            url = base_url.rstrip("/") + "/" + raw_value
    else:
        raise ValueError(f"Unsupported URL scheme for task image: {parsed.scheme!r}")

    request_kwargs: dict[str, Any] = {"timeout": timeout}
    if auth_header:
        request_kwargs["headers"] = {"Authorization": auth_header}
    if basic_auth:
        request_kwargs["auth"] = basic_auth

    response = requests.get(url, **request_kwargs)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        if response.status_code in (401, 403):
            raise ValueError(
                "Label Studio rejected image download with "
                f"HTTP {response.status_code}. Configure LABEL_STUDIO_AUTHORIZATION "
                "or LABEL_STUDIO_API_KEY for the inference service."
            ) from exc
        raise ValueError(
            f"Image download failed with HTTP {response.status_code} for {url!r}."
        ) from exc
    with Image.open(BytesIO(response.content)) as raw:
        return raw.convert("RGB")


def _looks_like_image_reference(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    parsed = urlparse(stripped)
    if parsed.scheme in ("http", "https", "file"):
        return True
    if stripped.startswith("/"):
        return True
    lower = stripped.lower()
    return lower.endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    )


def predictions_to_ls_result(
    aggregated_predictions,
    image_width: int,
    image_height: int,
    mapping: LabelStudioMapping,
    model_version: str,
) -> dict[str, Any]:
    """Convert aggregated predictions into a Label Studio result object.

    Pixel bounding boxes are translated into Label Studio percentage
    coordinates. Predictions whose ``category_id`` is not in the mapping
    are silently dropped so unsupported classes never leak to the UI.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image_width}x{image_height}")

    results: list[dict[str, Any]] = []
    scores: list[float] = []
    for prediction in aggregated_predictions:
        label_name = mapping.labels.get(int(prediction.category_id))
        if label_name is None:
            continue
        x0, y0, x1, y1 = prediction.bbox_xyxy
        x_percent = max(0.0, min(100.0, float(x0) / image_width * 100.0))
        y_percent = max(0.0, min(100.0, float(y0) / image_height * 100.0))
        width_percent = max(0.0, min(100.0 - x_percent, float(x1 - x0) / image_width * 100.0))
        height_percent = max(0.0, min(100.0 - y_percent, float(y1 - y0) / image_height * 100.0))
        if width_percent <= 0.0 or height_percent <= 0.0:
            continue
        score = float(prediction.score)
        scores.append(score)
        results.append(
            {
                "from_name": mapping.from_name,
                "to_name": mapping.to_name,
                "type": "rectanglelabels",
                "original_width": int(image_width),
                "original_height": int(image_height),
                "image_rotation": 0,
                "value": {
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rotation": 0,
                    "rectanglelabels": [label_name],
                },
                "score": score,
            }
        )

    aggregate_score = float(sum(scores) / len(scores)) if scores else 0.0
    return {
        "result": results,
        "score": aggregate_score,
        "model_version": model_version,
    }
