# CPU Performance and Evaluation Report

## O1 baseline (v0.3.0 source, 2026-11-16)

This is a local baseline for later CPU optimization comparisons. It is not an accuracy evaluation or a release performance claim. Both modes ran in separate Python processes against the same 100 development images, in the same deterministic order. The historical holdout was excluded.

### Measurement setup

- Source revision: `93ba16cb6c8d46564280b3459cdec731dcd2ef9f` (`v0.3.0`). The working-tree profiler is identified by SHA256 in each ignored run's `summary.json`.
- CPU: Intel Core i7-13620H; Windows 10 build 26200; Python 3.11.18; 2 PyTorch intra-op/inter-op and OpenCV threads; batch 1; CPU PyTorch backend.
- YOLO11n weights SHA256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`; confidence 0.25, IoU 0.45, max detections 300, image size 640.
- Fixed selection: 100 unique groups from `val/development`, seed 42, stratified over image size, object density, person/vehicle/traffic-light presence, scene, and time of day. The two final manifests have the same SHA256: `4baec873d34551f28e0d708f1cf62c4341304e4ce2092d06e484f6d7556b1f0d`.
- Each mode: 5 warmups, 3 sequential rounds of 100 samples, and API probes with 1/2/4 clients and 20 requests per client. API requests use an in-process ASGI transport, so they exclude socket/network/browser overhead. Busy timeout is 100 ms.
- Timing uses `perf_counter_ns`; process RSS is sampled every 50 ms. Nested timings overlap and must not be added together. API client time includes response JSON parsing; server total ends when the ASGI app returns after the response body send.
- No pipeline failures or non-200/503 API failures occurred. Model load time was 169 ms (`false`) and 187 ms (`true`); peak sampled RSS was 558.7 MiB and 551.8 MiB respectively.

### Results

| Path | include_detections | P50 | P95 | Three-round P95 median |
| --- | ---: | ---: | ---: | ---: |
| `run_single_image` pipeline (300 requests) | false | 42.08 ms | 47.12 ms | 47.18 ms |
| `run_single_image` pipeline (300 requests) | true | 52.15 ms | 58.85 ms | 58.85 ms |
| Local API successful requests (all phases) | false | 60.70 ms | 167.12 ms | 67.62 ms sequential |
| Local API successful requests (all phases) | true | 68.94 ms | 169.05 ms | 74.93 ms sequential |

The optional detections path adds about 10 ms to pipeline P50 and 11.67 ms to the median of round P95s (about 24.7%). Its observed work includes overlay details and PIL image opening. This is the current-state cost, not a comparison against an optimized candidate.

For API sequential rounds, each mode completed 300/300 requests without failures. In concurrency probes, `false` returned 20/20, 36/40, and 33/80 successful requests at 1/2/4 clients, with 0/4/47 busy 503s. `true` returned 20/20, 31/40, and 26/80, with 0/9/54 busy 503s. Successful-request latency percentiles exclude 503s. These are in-process saturation observations under a 100 ms busy timeout, not a network service capacity claim.

| Pipeline stage P50 | false | true |
| --- | ---: | ---: |
| OpenCV `imread` | 3.09 ms | 3.41 ms |
| YOLO preprocess | 0.98 ms | 1.16 ms |
| YOLO forward | 29.74 ms | 33.48 ms |
| Feature construction | 3.61 ms | 3.93 ms |
| Detection detail organization | not run | 4.60 ms |
| PIL open | not run | 0.30 ms |

Detector forward is the largest measured stage. `model_detect_ms` includes the detector stages and other runtime work; stage values overlap and should not be summed to infer end-to-end latency.

### Reproducibility and limits

The profiler now sorts feature tokens before greedy selection because separate processes may otherwise accumulate tiny floating-point differences in a different set iteration order and select different rows. Final manifests are byte-identical across modes. Per-request CSV, local performance manifest, full config/code/input hashes, and detailed summaries are kept in ignored `results/cpu-optimization/20261116-o1-finalA-{false,true}/` directories; the configs, including the local dataset path, remain under ignored `local/cpu-optimization/`.

The current `.venv` source is v0.3.0 while its installed distribution metadata says 0.2.2. The public v0.3.0 wheel was separately checked in O0, but this measured run uses the source checkout and local environment. The local ASGI API creates a second runtime within each process. Background Node/Python processes and available RAM were recorded in the O0 status; no user processes were stopped. RSS is sampled, so short peaks between 50 ms samples may be missed.

O1 provides the baseline needed by O2. It does not show a speedup, establish dataset independence for model quality, or satisfy any accuracy/recall gate. O2 should be compared on this exact fixed manifest and configuration before making a performance claim.

## O4 CPU backend comparison (2026-11-17)

The complete PyTorch/OpenVINO/ONNX Runtime matrix ran on the same 100-image development manifest. All optional backends were slower on at least one full pipeline/API path, none met the required 20% speedup, and each had exact output-feature mismatches despite detector boxes meeting the configured IoU/confidence tolerances. No optional backend is adopted. Full P95, 503, RSS, parity and gate results are in [`CPU_BACKEND_EXPERIMENT.md`](CPU_BACKEND_EXPERIMENT.md); raw ignored artifacts are in `results/cpu-optimization/backend-comparison-20261117-05/`.

## O2 shared-decode comparison (2026-11-16)

O2 reused the exact O1 sample manifest, configuration, CPU, checkpoint, and three-round workload. The implementation keeps the legacy path runtime available; production `ModelRuntime` now passes one EXIF-oriented BGR array to YOLO, feature construction, and optional box output.

| Path | Mode | O1 three-round P95 median | O2 three-round P95 median | Change |
| --- | --- | ---: | ---: | ---: |
| Pipeline | detections false | 47.18 ms | 43.45 ms | −7.9% |
| Pipeline | detections true | 58.85 ms | 41.92 ms | −28.7% |
| API sequential | detections false | 67.62 ms | 58.95 ms | −12.8% |
| API sequential | detections true | 74.93 ms | 59.29 ms | −20.9% |

O2 pipeline P50/P95 were 37.36/43.45 ms (`false`) and 38.23/43.22 ms (`true`). API sequential rounds each completed 300/300 requests. Under 1/2/4-client stress the `false` run returned 20/39/35 successful requests and 0/1/45 busy 503s; the `true` run returned 20/40/35 and 0/0/45. Pipeline failures remained zero. Sampled RSS was 555.1/549.0 MiB. Because this comparison uses one measured run per O2 mode and a local machine with background processes, report these as observed results on the fixed set, not a general speed guarantee.

The optional detections path no longer incurs a separate PIL dimension open on the production decoded runtime. For old path-only external runtimes, the compatibility branch still uses the legacy PIL dimensions behavior and is not included in the single-decode claim.

Parity evidence: 37 image I/O tests and the prescribed pipeline/API/build-features regression suite pass; across the same 100 development images, implementer-verified path/ndarray YOLO boxes matched within 1e-5 and public pipeline fields matched exactly. The BGR input array did not change during inference. The automated tests exercise EXIF orientations 1–8 in JPEG, PNG, and WebP, plus grayscale/RGB/RGBA, corruption, and the single-OpenCV-read contract. These checks establish parity only for the tested local set and runtime; they do not establish whole-dataset accuracy or model-quality improvement.

## CPU repair profiler revalidation (2026-11-17)

The repair profiler measured the v0.3.0 path from baseline commit `93ba16cb6c8d46564280b3459cdec731dcd2ef9f` against the shared-decode working tree on the same fixed 100-image development manifest (`4baec873d34551f28e0d708f1cf62c4341304e4ce2092d06e484f6d7556b1f0d`). The final comparison used three one-round repetitions per version and mode in a frozen AB/BA/AB sequence, with modes ordered false/true, true/false, false/true. Each task ran in a fresh process with seed 42, two threads, and one sequential pipeline/API measurement for every fixed sample. The sequence schedule SHA256 is `86164d11442006c82a07d3e39a3de96bce4e99775cf3a66670771475e602cb57`; its orchestration script and all task summaries are retained under ignored `results/cpu-repair/profile-ab-ba-ab/`.

| Path | Mode | v0.3.0 round P95s | v0.3.0 median | Shared-decode round P95s | Shared-decode median | Observed change |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Pipeline | detections false | 91.16, 100.64, 83.91 ms | 91.16 ms | 70.81, 172.85, 77.82 ms | 77.82 ms | −14.6% |
| Pipeline | detections true | 100.53, 96.08, 91.53 ms | 96.08 ms | 95.88, 83.46, 75.82 ms | 83.46 ms | −13.1% |
| API sequential | detections false | 108.76, 111.31, 116.48 ms | 111.31 ms | 83.74, 118.96, 96.42 ms | 96.42 ms | −13.4% |
| API sequential | detections true | 125.04, 117.14, 114.80 ms | 117.14 ms | 99.31, 102.00, 89.04 ms | 99.31 ms | −15.2% |

Each of the 12 paired tasks completed all 100 pipeline and 100 sequential API rows without a partial result. The one retained shared-decode false pipeline round had a 172.85 ms P95 and was not removed from the predeclared schedule. The profiler observed 100 legacy runtime calls and 200 OpenCV read calls per 100-image baseline pipeline round; shared decode observed 100 `detect_image` calls and 100 OpenCV reads. The source paths and SHA256s in each task summary identify the isolated baseline tree or current working tree actually imported. These measurements compare complete source snapshots on one workstation; treat the timing differences as local observations, not release guarantees.

The standard three-round-per-mode profiler reports also completed with 100 samples and `partial=false`; stage coverage identified actual detector preprocess, forward, NMS postprocess, feature, and hard-rule work. `httpx.ASGITransport` measures in-process API handling, including JSON parsing; it does not include network sockets, a browser, or deployment latency.
