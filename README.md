# Modal Deployment Guide — RF-DETR SAHI Label Studio Backend

This document describes how to deploy the FastAPI inference service to
Modal and how to connect it to a Label Studio project so predictions
become available as pre-annotations.

## 1. Prerequisites

- Active Modal account and local CLI authentication:
  ```bash
  uv venv
  source .venv/bin/activate
  uv pip install modal
  modal token new
  ```
- Deployment-ready checkpoints available under `models_deployment/`.
  The upload script reads the exact filenames defined in
  `modal_app/model_registry.py`. The required filenames are:
  `blue_gray_globules.pth`, `may_globules.pth`,
  `milia_like_cyst.pth`, `rosettes.pth`,
  `yellow_globules_ulcer.pth`.
- A Label Studio instance reachable from Modal's egress network. Auth
  is assumed to be disabled, so task image URLs are fetched anonymously.

## 2. Configuration

All settings are read from environment variables by
`modal_app/config.py`. Set them locally before running `modal deploy`
so they are captured into the Modal App environment, or attach them
through a `modal.Secret`.

This repository injects the Label Studio-related runtime variables into
the remote Modal container during deploy. In practice that means:

- `export SOME_VAR=...` in your terminal only affects the next
  `modal deploy` run; it is not magically visible to already deployed
  containers.
- after changing one of the runtime env vars below, redeploy the app.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODAL_APP_NAME` | `dermoscopy-rfdetr-sahi-inference` | Modal App name |
| `MODAL_ENVIRONMENT` | unset | Optional Modal environment |
| `MODAL_VOLUME_NAME` | `dermoscopy-rfdetr-checkpoints` | Volume name |
| `VOLUME_MOUNT_PATH` | `/models` | Container mount path |
| `MODAL_GPU` | `L4` | Modal GPU string; set to `cpu` or empty to disable GPU |
| `MODAL_REQUEST_TIMEOUT` | `900` | Per-request timeout in seconds |
| `MODEL_VERSION` | `rfdetr-sahi-v1` | Returned by `/health` and `/predict` |
| `TASK_IMAGE_FIELD` | `image` | Task-data key that points to the image asset |
| `LABEL_STUDIO_BASE_URL` | unset | Prefix for relative task image URLs; recommended when tasks contain `/data/upload/...` paths |
| `LABEL_STUDIO_AUTHORIZATION` | unset | Full HTTP `Authorization` header value used when downloading images from Label Studio |
| `LABEL_STUDIO_API_KEY` | unset | Convenience alternative to `LABEL_STUDIO_AUTHORIZATION`; combined with `LABEL_STUDIO_API_KEY_TYPE` |
| `LABEL_STUDIO_API_KEY_TYPE` | `Token` | Prefix used for `LABEL_STUDIO_API_KEY`, e.g. `Token` or `Bearer` |
| `LABEL_STUDIO_FROM_NAME` | `label` | Label Studio rectangle-control name |
| `LABEL_STUDIO_TO_NAME` | `image` | Label Studio image-tag name |
| `LABEL_STUDIO_MAPPING_JSON` | unset | Optional inline JSON for label mapping overrides |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.15` | Fallback per-class confidence |
| `DEFAULT_OVERLAP_FRACTION` | `0.25` | Request-level SAHI overlap |
| `DEFAULT_WBF_IOU_THRESHOLD` | `0.55` | Request-level WBF IoU |
| `IMAGE_REQUEST_TIMEOUT` | `30` | HTTP timeout when downloading task images |

### Optional label mapping override

Provide `LABEL_STUDIO_MAPPING_JSON` (for example through a Modal
Secret) to change the `from_name`, `to_name` or per-category labels:

```json
{
  "from_name": "label",
  "to_name": "image",
  "labels": {
    "1": "Blue-gray globules",
    "4": "MAY globules",
    "5": "Milia-like-cyst",
    "6": "Rosettes",
    "9": "Yellow globlues (ulcer)"
  }
}
```

The keys of `labels` are the `category_id` values from
`dermoscopy_dataset/result.json`. Any category id missing from the JSON
falls back to the default label defined in `model_registry.py`.

## 3. Upload checkpoints to the Modal Volume

From the repository root:

```bash
python -m modal_app.volume_bootstrap
```

Optional flags:

- `--volume-name` overrides `MODAL_VOLUME_NAME`.
- `--environment` overrides `MODAL_ENVIRONMENT`.
- `--models-root <dir>` resolves every checkpoint as
  `<dir>/<remote_filename>` instead of using the default
  `models_deployment/` directory.

The script creates the Volume when it does not yet exist, replaces the
currently served checkpoint files on each run, and exits non-zero if any
expected checkpoint is missing on disk.

## 4. Deploy the inference service

```bash
modal deploy -m modal_app.inference_service
```

The module form (`-m`) is required because `modal_app/inference_service.py`
uses relative imports (`from .config import ...`).

Modal prints the ASGI web endpoint after a successful deploy, for
example:

```
https://<workspace>--dermoscopy-rfdetr-sahi-inference-inferenceservice-fastapi-app.modal.run
```

Smoke test:

```bash
curl https://<deployed-host>/health
```

Expected response:

```json
{"status": "UP", "model_version": "rfdetr-sahi-v1"}
```

The first `/predict` call in a fresh container pays the RF-DETR model
load cost once; all subsequent calls reuse the cached models via
`@modal.enter()`.

## 5. Connect Label Studio

1. Open the target project in Label Studio.
2. Navigate to `Settings` → `Machine Learning` → `Add Model`.
3. Fill in:
   - `Title`: `RF-DETR SAHI`.
   - `URL`: the deployed Modal endpoint from section 4.
   - `Use for interactive preannotations`: enable as needed.
4. Save. Label Studio calls `GET /health` to validate the backend.
5. In the same settings page toggle
   `Retrieve predictions when loading a task` so predictions are fetched
   automatically when annotators open a task.
6. Open a task — bounding boxes produced by each class-specific model
   appear as pre-annotations ready for review.

### Label config requirement

The Label Studio labelling configuration must declare:

- a `RectangleLabels` control whose `name` matches
  `LABEL_STUDIO_FROM_NAME`,
- an `Image` tag whose `name` matches `LABEL_STUDIO_TO_NAME`,
- one `Label` entry per label string defined in the mapping.

Example using the default mapping:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="Blue-gray globules" background="#1f77b4"/>
    <Label value="MAY globules" background="#ff7f0e"/>
    <Label value="Milia-like-cyst" background="#2ca02c"/>
    <Label value="Rosettes" background="#d62728"/>
    <Label value="Yellow globlues (ulcer)" background="#9467bd"/>
  </RectangleLabels>
</View>
```

The `value` of every `<Label>` must match exactly the label name
returned by the backend.

### Optional: runtime defaults via *Any extra params* (Connect Model dialog)

The *Any extra params to pass during model connection* textarea in
Label Studio's *Connect Model* dialog is forwarded to the backend as
the `extra_params` field of the `POST /setup` request. The backend
accepts that field either as a parsed JSON object or as the raw JSON
string copied from the textarea, then parses it with the same schema as
the per-call `params` object from `POST /predict` and uses the result
as runtime defaults until the next `/setup` call.

Paste a JSON object in the same shape as the predict-request `params`:

```json
{
  "overlap_fraction": 0.3,
  "wbf_iou_threshold": 0.6,
  "per_class": {
    "blue_gray_globules":    {"enabled": true,  "patch_size": 128, "confidence_threshold": 0.20},
    "may_globules":          {"enabled": true,  "patch_size": 256, "confidence_threshold": 0.15},
    "milia_like_cyst":       {"enabled": true,  "patch_size": 256, "confidence_threshold": 0.30},
    "rosettes":              {"enabled": false, "patch_size": 320, "confidence_threshold": 0.10},
    "yellow_globlues_ulcer": {"enabled": true,  "patch_size": 512, "confidence_threshold": 0.25}
  }
}
```

Precedence at prediction time:

1. `params` from the per-call `/predict` request body,
2. runtime defaults stored from the last `/setup` call,
3. environment defaults / training-time `patch_size` from the registry.

Caveats:

- The runtime defaults live in the FastAPI process memory of the Modal
  container that received `/setup`. They are **not shared** with other
  replicas; if you scale past one replica add an external store (for
  example `modal.Dict`).
- A container cold start empties the runtime defaults. Label Studio
  re-triggers `/setup` whenever the ML backend page is saved or
  reconnected; clicking *Validate and Save* in the dialog restores them.
- Invalid JSON or unknown class keys cause `/setup` to fail with HTTP
  400; Label Studio surfaces the error next to *Validate and Save*.
- An empty textarea, missing request body, or empty `extra_params`
  clears the runtime defaults (the backend falls back to environment /
  registry defaults again).
- The backend derives a prediction-specific `model_version` fingerprint
  from the effective inference parameters. This helps Label Studio
  distinguish predictions generated with different SAHI settings.

## 6. Request overrides

`POST /predict` accepts per-class SAHI `enabled`, `patch_size` and
`confidence_threshold` overrides in a single request payload:

```json
{
  "tasks": [
    {"id": 42, "data": {"image": "http://localhost:8080/data/upload/42.jpg"}}
  ],
  "params": {
    "overlap_fraction": 0.3,
    "wbf_iou_threshold": 0.6,
    "per_class": {
      "blue_gray_globules":    {"enabled": true, "patch_size": 128, "confidence_threshold": 0.2},
      "yellow_globlues_ulcer": {"enabled": false, "patch_size": 512, "confidence_threshold": 0.25}
    }
  }
}
```

Fields are optional. Missing per-class entries fall back to each
model's training-time `patch_size` and to
`DEFAULT_CONFIDENCE_THRESHOLD`. Missing `enabled` falls back to `true`.
Unknown class keys are rejected with HTTP 400.

When Label Studio itself calls `/predict`, it may also include
transport-level keys inside `params` such as `context`, `login`, and
`password`. The backend ignores those fields and validates only the
SAHI-specific overrides shown above.

