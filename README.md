# Image-Based Traffic Risk Assessment

> **v0.1.0 withdrawn:** the historical model release is retained for audit but is not downloadable or loadable. See [docs/MODELS.md](docs/MODELS.md) for the provenance and contract limitations. No BDD100K-derived weights are redistributed by this source-only fix.

An open research and teaching demo that estimates a coarse scene-risk category from a single road image. The pipeline combines YOLO object detection, a traffic-light colour classifier, structured scene features, deterministic rules, and an XGBoost classifier behind a FastAPI API and React interface.

> [!CAUTION]
> This project is **not a safety system**. Its output is not a collision probability, driving instruction, or validated measure of real-world danger. Do not use it for vehicle control, traffic enforcement, emergency response, or any other safety-critical decision.

## What this release provides

- Reproducible CPU-first inference with separately downloaded, checksum-verified model assets.
- Explicit `unknown` results when perception or model output is insufficient.
- A versioned feature schema shared by inference and model loading.
- A hardened upload API with bounded image size and structured errors.
- Unit, API, and frontend tests that run without downloading large models.

The included risk model was trained from **weak labels derived from heuristics**. Any reported evaluation against those labels is a *weak-label agreement measurement*, not evidence of real-world traffic-risk accuracy. See [MODEL_CARD.md](MODEL_CARD.md) for limitations.

## Quick start (CPU)

Requirements: Python 3.11, Node.js 20.19+ (or 22.12+) for the optional web UI, and Git.

```bash
git clone https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment.git
cd Image_Based-Traffic-Risk-Assessment
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

On Windows or Linux, install the shared CPU lock, then retrieve the release models:

```bash
python -m pip install --upgrade pip
python -m pip install --no-deps -r requirements-cpu.lock
python -m pip install --no-deps -e .
python -m tools.download_models
```

`requirements-cpu.lock` is the fully resolved Python 3.11 runtime dependency set shared by Windows and Linux CI. It uses the official PyTorch CPU wheel index, the official CPU-only `xgboost-cpu==3.2.0` distribution, and the single OpenCV distribution required by both this project and Ultralytics. Install the lock and editable project with `--no-deps` to preserve that exact resolved environment.

The shared CPU lock does not target macOS or other platforms. There, install from project metadata instead; the platform marker selects the official full `xgboost==3.2.0` distribution while preserving the same `import xgboost` API:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip check
python -m tools.download_models
```

The downloader reads its packaged, versioned asset manifest and refuses files whose size or SHA-256 does not match. The runtime repeats exact size and SHA-256 verification for all six assets before loading any checkpoint. Model binaries, BDD100K images, generated predictions, and training outputs are intentionally excluded from Git. See [docs/MODELS.md](docs/MODELS.md) for the asset trust and compatibility rules.

Run one image through the command-line pipeline:

```bash
python -m src.pipeline --img path/to/image.jpg
```

Start the API:

```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/ready` before sending inference requests. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

For NVIDIA acceleration, install a PyTorch build matching your driver before installing this project; do not mix CUDA wheel indexes into the default CPU installation. See [docs/CUDA.md](docs/CUDA.md).

## API contract

Submit a JPEG, PNG, or WebP image as multipart field `file`. The filename must use an allowed `.jpg`, `.jpeg`, `.png`, or `.webp` extension; the server also verifies the declared MIME type, file signature, and decoded image format independently:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@path/to/image.jpg"
```

A successful inference returns HTTP 200:

```json
{
  "status": "ok",
  "risk": "medium",
  "image_id": "c6d0d74c-9af3-4d31-8f9e-51f9ca7b9380",
  "reason": "model prediction",
  "features": {},
  "model_versions": {}
}
```

An image that cannot support a defensible estimate also returns HTTP 200, but never disguises uncertainty as low risk:

```json
{
  "status": "uncertain",
  "risk": "unknown",
  "image_id": "c6d0d74c-9af3-4d31-8f9e-51f9ca7b9380",
  "reason": "perception produced no usable detections",
  "features": {},
  "model_versions": {}
}
```

Input errors use 400/413/415/422, unavailable or busy inference uses 503, and unexpected internal errors use 500. Error bodies are structured and do not expose local paths. `/health` reports process liveness; `/ready` reports whether all required models and the feature schema are loaded.

Default upload limits are 10 MiB and 20 megapixels. Default CORS access is `http://localhost:5173` without credentials. Runtime configuration is read at process startup:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RISK_MODEL_DIR` | `./models` | Root of downloaded model assets |
| `RISK_UPLOAD_DIR` | system temp + `traffic-risk-assessment/uploads` | Temporary upload directory |
| `RISK_MAX_UPLOAD_BYTES` | `10485760` | Maximum uploaded bytes |
| `RISK_MAX_IMAGE_PIXELS` | `20000000` | Maximum decoded width Ã— height |
| `RISK_INFERENCE_CONCURRENCY` | `1` | Maximum simultaneous inference jobs |
| `RISK_BUSY_TIMEOUT_SECONDS` | `0.1` | Wait before returning busy/503 |
| `RISK_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `RISK_DEVICE` | `cpu` | Torch device such as `cpu` or `cuda:0` |

## Web interface

With the API running:

```bash
cd risk_frontend
npm ci
npm run dev
```

The development server proxies API requests by default. Set `VITE_API_BASE_URL` when the API is hosted elsewhere.

## Data and training

This repository does **not** redistribute BDD100K images or annotations. Obtain the dataset from its official provider and accept the applicable data terms yourself. The BSD-3-Clause license of the BDD100K tooling repository does not automatically grant redistribution rights for the dataset. See [docs/DATA.md](docs/DATA.md).

Training utilities are retained for research, but v0.1.0 guarantees inference reproducibility only. Reproducing model training requires independently obtained data, the documented schema, fixed seeds, and review of the weak-label process.

Install the optional training dependencies with `python -m pip install -e ".[training]"`. The training extra includes scikit-learn and joblib for historical data utilities, but runtime model loading accepts only the verified native XGBoost UBJ artifact.

## Development

```bash
python -m pip install --no-deps -r requirements-dev.lock
python -m pip install --no-deps -e .
python -m compileall -q src tests tools
ruff check src tests tools
pytest --cov=src --cov-report=term-missing --cov-fail-under=80

cd risk_frontend
npm ci
npm run lint
npm test
npm run build
```

Ordinary CI uses mocks and synthetic images and does not download release models. A separate manual smoke workflow may exercise real models.

## Project layout

```text
src/                  inference, feature extraction, rules, and FastAPI service
tools/                model download and supporting command-line utilities
tests/                Python unit and API tests
risk_frontend/        React/Vite user interface
docs/                 model, data, and accelerator setup notes
.github/workflows/    cross-platform tests and security checks
```

## License and attribution

Copyright (C) 2026 Bowen

This repository is licensed under **GNU Affero General Public License v3.0 only** (`AGPL-3.0-only`). See [LICENSE](LICENSE). Network users must be offered the corresponding source as required by section 13.

Third-party packages, models, and datasets retain their own terms. In particular, the project uses Ultralytics software/model assets under its AGPL-3.0 option. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Contributions are welcome under the same license. Read [CONTRIBUTING.md](CONTRIBUTING.md), cite the versioned software using [CITATION.cff](CITATION.cff), and report vulnerabilities as described in [SECURITY.md](SECURITY.md).

