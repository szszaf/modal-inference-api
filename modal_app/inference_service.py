"""Modal + FastAPI inference service for Label Studio.

The module defines a single Modal App exposing one class that:

* mounts the Modal Volume that holds RF-DETR checkpoints (read-only),
* loads every supported class-specific model on container startup and
  keeps them cached for the container's lifetime,
* serves a FastAPI ASGI application with Label Studio ML backend
  endpoints (``/health``, ``/setup``, ``/predict``) plus a multipart
  upload endpoint (``/predict-file``) for local files.

All SAHI patching and weighted boxes fusion logic is reused from
``pipeline/inference.py`` and ``pipeline/aggregation.py`` without
modification.
"""

import hashlib
import json
import os
from io import BytesIO

import modal

from .config import get_modal_settings, resolve_gpu


_settings = get_modal_settings()

_REMOTE_ENV_KEYS = (
    "MODEL_VERSION",
    "TASK_IMAGE_FIELD",
    "LABEL_STUDIO_BASE_URL",
    "LABEL_STUDIO_AUTHORIZATION",
    "LABEL_STUDIO_API_KEY",
    "LABEL_STUDIO_API_KEY_TYPE",
    "LABEL_STUDIO_MAPPING_JSON",
    "LABEL_STUDIO_FROM_NAME",
    "LABEL_STUDIO_TO_NAME",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_OVERLAP_FRACTION",
    "DEFAULT_WBF_IOU_THRESHOLD",
    "IMAGE_REQUEST_TIMEOUT",
)

_runtime_secret = modal.Secret.from_dict(
    {key: os.environ.get(key) for key in _REMOTE_ENV_KEYS}
)


_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("pipeline", "modal_app")
)


_volume = modal.Volume.from_name(
    _settings.volume_name,
    environment_name=_settings.modal_environment,
    create_if_missing=False,
)


app = modal.App(name=_settings.modal_app_name)


@app.cls(
    image=_image,
    gpu=resolve_gpu(_settings.gpu_type),
    volumes={_settings.volume_mount_path: _volume},
    timeout=_settings.request_timeout_seconds,
    secrets=[_runtime_secret],
    max_containers=3,
)
class InferenceService:
    @modal.enter()
    def load_models(self) -> None:
        from pipeline.inference import load_rfdetr_model

        from .model_registry import MODEL_REGISTRY, remote_checkpoint_mount_path

        self._models: dict[str, object] = {}
        for entry in MODEL_REGISTRY:
            checkpoint_path = remote_checkpoint_mount_path(_settings.volume_mount_path, entry)
            print(
                f"[inference-service] loading class_key={entry.class_key} "
                f"variant={entry.variant} checkpoint={checkpoint_path}"
            )
            self._models[entry.class_key] = load_rfdetr_model(
                model_variant=entry.variant,
                checkpoint_path=checkpoint_path,
            )
        print(f"[inference-service] loaded {len(self._models)} models")

    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from PIL import Image, UnidentifiedImageError
        from pydantic import ValidationError

        from pipeline.aggregation import apply_wbf, filter_by_score
        from pipeline.inference import run_sahi_inference

        from .label_studio_adapter import (
            PredictionParams,
            PredictRequest,
            PredictRequestParams,
            SetupRequest,
            derive_label_studio_base_url,
            first_non_none,
            load_label_studio_mapping,
            parse_setup_extra_params,
            predictions_to_ls_result,
            resolve_image,
        )
        from .model_registry import (
            MODEL_REGISTRY,
            SUPPORTED_CLASS_KEYS,
            default_label_for_category_id,
        )

        settings = _settings
        models = self._models
        registry = MODEL_REGISTRY

        mapping = load_label_studio_mapping(
            mapping_json=settings.label_studio_mapping_json,
            default_from_name=settings.label_studio_from_name,
            default_to_name=settings.label_studio_to_name,
            default_labels=default_label_for_category_id(),
        )

        # Runtime defaults delivered by Label Studio through /setup's
        # extra_params field. Wrapped in a dict so the FastAPI handlers
        # below can mutate it through the closure.
        runtime_state: dict[str, object | None] = {
            "defaults": None,
            "label_studio_base_url": derive_label_studio_base_url(
                settings.label_studio_base_url,
                None,
            ),
        }

        def _reject_unknown_class_keys(per_class_overrides: dict) -> None:
            unknown_keys = set(per_class_overrides.keys()) - SUPPORTED_CLASS_KEYS
            if unknown_keys:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Unknown class keys in per_class overrides.",
                        "unknown": sorted(unknown_keys),
                        "supported": sorted(SUPPORTED_CLASS_KEYS),
                    },
                )

        def _effective_global_overlap_fraction(per_call, runtime) -> float:
            return first_non_none(
                per_call.overlap_fraction if per_call is not None else None,
                runtime.overlap_fraction if runtime is not None else None,
                settings.default_overlap_fraction,
            )

        def _effective_global_wbf_iou_threshold(per_call, runtime) -> float:
            return first_non_none(
                per_call.wbf_iou_threshold if per_call is not None else None,
                runtime.wbf_iou_threshold if runtime is not None else None,
                settings.default_wbf_iou_threshold,
            )

        def _effective_class_overlap_fraction(
            call_override,
            runtime_override,
            per_call,
            runtime,
        ) -> float:
            return first_non_none(
                call_override.overlap_fraction if call_override is not None else None,
                per_call.overlap_fraction if per_call is not None else None,
                runtime_override.overlap_fraction if runtime_override is not None else None,
                runtime.overlap_fraction if runtime is not None else None,
                settings.default_overlap_fraction,
            )

        def _effective_class_wbf_iou_threshold(
            call_override,
            runtime_override,
            per_call,
            runtime,
        ) -> float:
            return first_non_none(
                call_override.wbf_iou_threshold if call_override is not None else None,
                per_call.wbf_iou_threshold if per_call is not None else None,
                runtime_override.wbf_iou_threshold if runtime_override is not None else None,
                runtime.wbf_iou_threshold if runtime is not None else None,
                settings.default_wbf_iou_threshold,
            )

        def _prediction_model_version(
            per_call,
            runtime,
            per_call_per_class: dict,
            runtime_per_class: dict,
        ) -> str:
            effective_params = {
                "overlap_fraction": float(
                    _effective_global_overlap_fraction(per_call, runtime)
                ),
                "wbf_iou_threshold": float(
                    _effective_global_wbf_iou_threshold(per_call, runtime)
                ),
                "per_class": {},
            }
            for entry in registry:
                call_override = per_call_per_class.get(entry.class_key)
                runtime_override = runtime_per_class.get(entry.class_key)
                effective_params["per_class"][entry.class_key] = {
                    "enabled": bool(
                        first_non_none(
                            call_override.enabled if call_override is not None else None,
                            runtime_override.enabled if runtime_override is not None else None,
                            True,
                        )
                    ),
                    "patch_size": int(
                        first_non_none(
                            call_override.patch_size if call_override is not None else None,
                            runtime_override.patch_size if runtime_override is not None else None,
                            entry.training_patch_size,
                        )
                    ),
                    "confidence_threshold": float(
                        first_non_none(
                            call_override.confidence_threshold if call_override is not None else None,
                            runtime_override.confidence_threshold if runtime_override is not None else None,
                            settings.default_confidence_threshold,
                        )
                    ),
                    "overlap_fraction": float(
                        _effective_class_overlap_fraction(
                            call_override,
                            runtime_override,
                            per_call,
                            runtime,
                        )
                    ),
                    "wbf_iou_threshold": float(
                        _effective_class_wbf_iou_threshold(
                            call_override,
                            runtime_override,
                            per_call,
                            runtime,
                        )
                    ),
                }

            fingerprint = hashlib.sha256(
                json.dumps(effective_params, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()[:10]
            return f"{settings.model_version}__cfg-{fingerprint}"

        def _parse_form_params(params: str | None) -> PredictRequestParams | None:
            if params is None or params.strip() == "":
                return None
            try:
                raw = json.loads(params)
            except json.JSONDecodeError as error:
                raise HTTPException(
                    status_code=400,
                    detail=f"params must be valid JSON: {error}",
                ) from error
            if not isinstance(raw, dict):
                raise HTTPException(
                    status_code=400,
                    detail="params JSON must decode to an object.",
                )
            try:
                parsed = PredictRequestParams.model_validate(raw)
            except ValidationError as error:
                raise HTTPException(status_code=400, detail=error.errors()) from error
            _reject_unknown_class_keys(parsed.per_class)
            return parsed

        def _predict_loaded_images(
            images: list[tuple[int | str | None, object]],
            per_call: PredictRequestParams | None,
        ) -> dict:
            per_call_per_class = per_call.per_class if per_call is not None else {}
            _reject_unknown_class_keys(per_call_per_class)

            runtime = runtime_state["defaults"]
            runtime_per_class = runtime.per_class if runtime is not None else {}

            prediction_model_version = _prediction_model_version(
                per_call=per_call,
                runtime=runtime,
                per_call_per_class=per_call_per_class,
                runtime_per_class=runtime_per_class,
            )
            print(
                "[inference-service] predict effective version: "
                f"{prediction_model_version}"
            )

            task_results: list[dict] = []
            for task_id, pil_image in images:
                image_width, image_height = pil_image.size
                merged_predictions: list = []

                for entry in registry:
                    call_override = per_call_per_class.get(entry.class_key)
                    runtime_override = runtime_per_class.get(entry.class_key)
                    enabled = bool(
                        first_non_none(
                            call_override.enabled if call_override is not None else None,
                            runtime_override.enabled if runtime_override is not None else None,
                            True,
                        )
                    )
                    if not enabled:
                        print(
                            "[inference-service] skipping disabled class during predict: "
                            f"task_id={task_id!r} class_key={entry.class_key}"
                        )
                        continue
                    patch_size = first_non_none(
                        call_override.patch_size if call_override is not None else None,
                        runtime_override.patch_size if runtime_override is not None else None,
                        entry.training_patch_size,
                    )
                    confidence_threshold = first_non_none(
                        call_override.confidence_threshold if call_override is not None else None,
                        runtime_override.confidence_threshold if runtime_override is not None else None,
                        settings.default_confidence_threshold,
                    )
                    class_overlap_fraction = _effective_class_overlap_fraction(
                        call_override,
                        runtime_override,
                        per_call,
                        runtime,
                    )
                    class_wbf_iou_threshold = _effective_class_wbf_iou_threshold(
                        call_override,
                        runtime_override,
                        per_call,
                        runtime,
                    )

                    inference_result = run_sahi_inference(
                        model=models[entry.class_key],
                        image=pil_image,
                        patch_size=patch_size,
                        overlap_fraction=class_overlap_fraction,
                        score_threshold=confidence_threshold,
                        category_id_override=entry.category_id,
                    )
                    filtered = filter_by_score(
                        inference_result.predictions, confidence_threshold
                    )
                    aggregated = apply_wbf(
                        filtered,
                        image_width=inference_result.image_width,
                        image_height=inference_result.image_height,
                        iou_threshold=class_wbf_iou_threshold,
                    )
                    merged_predictions.extend(aggregated)

                task_results.append(
                    predictions_to_ls_result(
                        aggregated_predictions=merged_predictions,
                        image_width=image_width,
                        image_height=image_height,
                        mapping=mapping,
                        model_version=prediction_model_version,
                    )
                )

            return {
                "results": task_results,
                "model_version": prediction_model_version,
            }

        api = FastAPI(
            title="RF-DETR SAHI Label Studio Backend",
            version=settings.model_version,
        )

        @api.get("/health")
        def health() -> dict[str, str]:
            return {"status": "UP", "model_version": settings.model_version}

        @api.post("/setup")
        def setup(request: SetupRequest | None = None) -> dict[str, str]:
            try:
                extras = parse_setup_extra_params(
                    request.extra_params if request is not None else None
                )
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            if extras:
                try:
                    parsed = PredictionParams.model_validate(extras)
                except ValidationError as error:
                    raise HTTPException(
                        status_code=400, detail=error.errors()
                    ) from error
                _reject_unknown_class_keys(parsed.per_class)
                runtime_state["defaults"] = parsed
                print(
                    "[inference-service] runtime defaults updated from /setup: "
                    f"{parsed.model_dump(exclude_none=True)}"
                )
            else:
                runtime_state["defaults"] = None
                print("[inference-service] runtime defaults cleared (empty extra_params)")

            resolved_base_url = derive_label_studio_base_url(
                settings.label_studio_base_url,
                request.hostname if request is not None else None,
            )
            runtime_state["label_studio_base_url"] = resolved_base_url
            print(
                "[inference-service] label studio base url resolved: "
                f"configured={settings.label_studio_base_url!r} "
                f"setup_hostname={request.hostname if request is not None else None!r} "
                f"effective={resolved_base_url!r}"
            )
            return {"model_version": settings.model_version}

        @api.post("/predict")
        def predict(request: PredictRequest) -> dict:
            per_call = request.params
            loaded_images = []
            for task in request.tasks:
                try:
                    pil_image = resolve_image(
                        task_data=task.data,
                        task_image_field=settings.task_image_field,
                        base_url=runtime_state["label_studio_base_url"],
                        timeout=settings.image_request_timeout_seconds,
                        auth_header=settings.label_studio_auth_header,
                        basic_auth=(
                            (per_call.login, per_call.password)
                            if per_call is not None
                            and per_call.login is not None
                            and per_call.password is not None
                            else None
                        ),
                    )
                except (KeyError, TypeError, ValueError) as error:
                    print(
                        "[inference-service] predict image-resolution error: "
                        f"task_id={task.id!r} "
                        f"task_image_field={settings.task_image_field!r} "
                        f"available_keys={sorted(task.data.keys())} "
                        f"raw_value={task.data.get(settings.task_image_field)!r} "
                        f"detail={error}"
                    )
                    raise HTTPException(status_code=400, detail=str(error)) from error
                loaded_images.append((task.id, pil_image))
            return _predict_loaded_images(loaded_images, per_call)

        @api.post("/predict-file")
        async def predict_file(
            image: UploadFile = File(...),
            params: str | None = Form(default=None),
            task_id: str | None = Form(default=None),
        ) -> dict:
            per_call = _parse_form_params(params)
            content = await image.read()
            try:
                with Image.open(BytesIO(content)) as raw:
                    pil_image = raw.convert("RGB")
            except (UnidentifiedImageError, OSError) as error:
                raise HTTPException(
                    status_code=400,
                    detail=f"Uploaded file is not a supported image: {error}",
                ) from error

            resolved_task_id = task_id or image.filename
            return _predict_loaded_images([(resolved_task_id, pil_image)], per_call)

        return api
