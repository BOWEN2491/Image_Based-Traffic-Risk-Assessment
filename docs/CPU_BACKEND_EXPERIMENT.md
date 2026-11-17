# CPU Backend Experiment

## Result

The complete O4 matrix finished on 2026-11-17 in 957 seconds. All six PyTorch, OpenVINO and ONNX Runtime CPU configurations completed against the fixed 100-image development manifest. Neither optional backend met the parity and performance adoption gates; keep the default PyTorch backend and do not expose a new backend setting.

This is a same-checkpoint engineering comparison on one Windows CPU environment. It is not a detector quality evaluation, a cross-platform claim, or evidence that the detector is accurate on natural road data.

## Frozen inputs and execution

### Recreating the isolated backend environment

Create a fresh Python 3.11 environment, install `requirements-cpu.lock`, then install `requirements-cpu-backend.txt`. The latter pins only the optional experiment packages; it does not change the default runtime lock. Copy `tools/config_templates/compare_cpu_backends.example.json`, then set the local development dataset root, the frozen 100-image performance manifest, checkpoint path/hash, and the four verified artifact-manifest paths. Keep all paths local to the workspace or absolute on the workstation, and write each run to a new output directory.

- Verified experiment versions: Python 3.11.18, NumPy 2.3.5, PyTorch 2.13.0+cpu, Ultralytics 8.4.126, OpenVINO 2025.4.1, ONNX 1.23.0, and ONNX Runtime 1.32.0. NumPy is pinned separately because the default CPU lock uses 2.4.6 while the isolated backend executor requires 2.3.5.
- The example config intentionally contains placeholders for the dataset, manifest, and checkpoint hash. The manifest must contain exactly the frozen 100 development rows used by the experiment; do not regenerate or resample it.
- The repair runner writes schema version 2 and expects 920 API rows per configuration (40 warmup, 600 sequential across both modes, 280 concurrency) plus 600 measured pipeline rows across both include-detections modes. Each mode contributes 300 pipeline rows and 300 sequential API rows. Historical schema-v1 runs remain unchanged.

- Source model: public YOLO11n; SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Fixed performance manifest: 100 unique development source groups, manifest SHA256 `4baec873d34551f28e0d708f1cf62c4341304e4ce2092d06e484f6d7556b1f0d`. It is not a final quality holdout.
- Platform: Windows 10 build 26200, Python 3.11.18, Intel Core i7-13620H; batch 1, seed 42, two CPU threads; 5 warmups and 3 rounds per path.
- All configurations used FP32, confidence 0.25, NMS IoU 0.45, max 300 detections, square `rect=False` input, and verified actual shapes `[1,3,640,640]` or `[1,3,960,960]`. PyTorch at each size is the square preprocessing reference; this does not assert equivalence to production O2's rectangular preprocessing.
- Optional backend packages were isolated from the default environment. The actual providers reported CPU-only execution. Exported OpenVINO/ONNX assets have independent manifests with source checkpoint identity, bundle file hashes, static dimensions, FP32 verification, and class mapping.
- The full raw output, per-request CSV, parity JSONL, run configuration, and hashes are in ignored `results/cpu-optimization/backend-comparison-20261117-05/`. The prior partial runs are retained and excluded from conclusions.

## Timing

Each cell reports the median of the three per-round P95s in milliseconds, first the measured value and then its ratio to same-size square PyTorch. API P95 includes successful responses only; HTTP 503 busy rejections are counted separately.

| Backend | Size | Pipeline P95 false | Pipeline P95 true | API P95 false | API P95 true | Peak RSS ratio | API 503 false/true | Adoption |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch reference | 640 | 61.18 | 71.80 | 82.74 | 90.86 | 1.000 | 73 / 77 | Reference |
| OpenVINO | 640 | 73.76 (1.206×) | 74.04 (1.031×) | 83.94 (1.014×) | 95.36 (1.049×) | 1.110× | 74 / 78 | `not_adopted` |
| ONNX Runtime | 640 | 75.24 (1.230×) | 74.29 (1.035×) | 96.07 (1.161×) | 95.57 (1.052×) | 0.966× | 70 / 75 | `not_adopted` |
| PyTorch reference | 960 | 144.29 | 155.40 | 165.03 | 164.15 | 1.000 | 86 / 86 | Reference |
| OpenVINO | 960 | 170.96 (1.185×) | 173.54 (1.117×) | 204.79 (1.241×) | 197.36 (1.202×) | 0.945× | 92 / 91 | `not_adopted` |
| ONNX Runtime | 960 | 181.83 (1.260×) | 166.80 (1.073×) | 171.19 (1.037×) | 200.02 (1.218×) | 0.899× | 89 / 87 | `not_adopted` |

Each configuration completed 300 pipeline requests per detections mode and 890 API request rows overall (445 per mode: 5 warmups, 300 sequential requests, and 140 concurrency requests). Sequential API requests were 300/300 HTTP 200 per mode. Pipeline failures were zero. API successful-request percentiles exclude 503s; the table preserves each mode's 503 count.

No candidate achieved the required 20% improvement in both full pipeline and API paths for both detections modes. Each candidate also regressed beyond 5% in at least one path. Peak RSS stayed within the 1.25× gate. Busy 503 counts increased versus PyTorch for OpenVINO at both sizes and ONNX at 960.

## Output parity

Detector boxes for all 100 images at both sizes passed the one-to-one matching thresholds of IoU ≥0.99 and absolute confidence difference ≤0.001, with no unmatched boxes. Across 200 response rows per backend/size (both detections modes), exact public-field parity failed:

| Backend | Size | Response rows with feature differences | `high↔unknown` transitions | Parity gate |
| --- | ---: | ---: | ---: | --- |
| OpenVINO | 640 | 178 / 200 | 0 | fail |
| ONNX Runtime | 640 | 176 / 200 | 0 | fail |
| OpenVINO | 960 | 182 / 200 | 0 | fail |
| ONNX Runtime | 960 | 184 / 200 | 0 | fail |

The mismatched response field was `features`; `status`, `risk`, `reason_code`, `reason`, and `image_size` matched, with no `high↔unknown` change. An independent offline recheck of the saved `parity_outputs.jsonl` found that the numeric geometry feature differences were at most `4.6568e-7`; all were within the separately declared `5e-7` diagnostic tolerance, and the seven feature-level maxima are recorded in `offline_analysis.json`. This does not change the plan's strict exact-feature parity gate: the gate remains failed, no tolerances were widened after seeing results, and no backend is adopted.

## Gate and follow-up

- Backend parity: `fail` for all four optional backend/size pairs due to exact feature differences.
- Performance adoption: `not_adopted` for all four pairs; none met the 20% improvement and 5% non-regression requirements.
- Memory: all four optional configurations passed the 1.25× RSS ceiling.
- Quality promotion: blocked by O3. The development manifest cannot substitute for a new independently sourced, human-reviewed holdout.
- A worker accounting bug in the first complete measurement run incorrectly expected 1,500 API rows; the corrected run expects 890 and verifies sequential success separately for each mode. Four synthetic tests cover this accounting and adoption gate behavior.
- After the final run, the saved CSVs, worker summaries, artifact manifests and hashes were re-read without inference using `local/cpu-optimization/analyze_backend_run.py`; its results are stored as `offline_analysis.json` alongside the original `comparison.json`.
- Revisit optional backends only with a separately justified implementation change and the same predeclared gates. Keep the shipped default PyTorch 640 configuration unchanged.

## CPU repair round-robin revalidation (2026-11-17)

The repair runner completed a new schema-v2 run in 1,245.7 seconds. It used 18 serial fresh-process measurement tasks followed by six serial concurrency probes. All 24 tasks completed; concurrency began only after all formal rounds completed. The frozen schedule SHA256 is `93f38cd59ae1383515d10424b97872f13b242fa49d4b72d41d3c7eab7920416c`, and the fixed 100-image development manifest SHA256 remains `4baec873d34551f28e0d708f1cf62c4341304e4ce2092d06e484f6d7556b1f0d`. Full task logs, round outputs, CSV/JSONL, schedule, manifests, and summaries are retained in ignored `results/cpu-repair/backend-matrix-final-20261117/`; the historical O4 run above was not modified.

Each config contains 600 formal pipeline rows across both detections modes (300 per mode), 600 successful sequential API requests across both modes (300 per mode), 40 API warmups, and 280 concurrency rows, for 920 API rows total. Every config passed exact identity, input-hash, provider, round-output, and 400-comparison round-stability checks. Total elapsed time was 1,245.7 seconds; cumulative per-config time ranged from 141.3 to 300.2 seconds, below the 900-second cap. Cold load and warmup are excluded from formal P95; round values below are the median of each config's three measured per-round P95s.

| Backend | Size | Pipeline P95 false / true | API P95 false / true | P95 ratios vs same-size Torch (pipeline false / true; API false / true) | Concurrency 503 false / true | Exact parity | Adoption |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| PyTorch | 640 | 94.00 / 93.72 ms | 118.16 / 118.01 ms | 1.000 / 1.000; 1.000 / 1.000 | 70 / 78 | Reference | Reference |
| OpenVINO | 640 | 98.31 / 92.69 ms | 102.54 / 114.30 ms | 1.046 / 0.989; 0.868 / 0.969 | 79 / 76 | fail (178 rows) | `not_adopted` |
| ONNX Runtime | 640 | 112.42 / 114.72 ms | 134.47 / 139.06 ms | 1.196 / 1.224; 1.138 / 1.178 | 59 / 73 | fail (176 rows) | `not_adopted` |
| PyTorch | 960 | 191.98 / 184.85 ms | 193.95 / 198.27 ms | 1.000 / 1.000; 1.000 / 1.000 | 83 / 86 | Reference | Reference |
| OpenVINO | 960 | 203.34 / 195.71 ms | 227.33 / 214.95 ms | 1.059 / 1.059; 1.172 / 1.084 | 87 / 88 | fail (182 rows) | `not_adopted` |
| ONNX Runtime | 960 | 240.95 / 244.27 ms | 266.06 / 272.03 ms | 1.255 / 1.321; 1.372 / 1.372 | 86 / 87 | fail (184 rows) | `not_adopted` |

All detector box sets passed IoU ≥0.99 and confidence difference ≤0.001; no high/unknown transitions occurred. Exact public feature parity still failed for each optional backend. All configurations passed the 1.25× RSS ceiling and had zero pipeline failures; the predeclared failure-rate gate failed for OpenVINO at both sizes and ONNX at 960. None met the required 20% pipeline and API speedup in both modes, so the default remains PyTorch and all optional backends remain `not_adopted`.
