# CPU perception experiments: maintenance release 0.2.2

The public default remains YOLO11n with `high/unknown` rules. Neither candidate is promoted. YOLO11s improves local detection metrics but fails the repeated CPU latency limit; MobileNetV3-Small is faster on crops but does not improve holdout color F1. Independently, local annotation provenance is unverified and the yellow-light holdout is too small for a supported color-upgrade claim.

## Data and measurement boundary

- Seed 42; fixed source/ride-grouped split; 1,000 validation images: 502 development and 498 holdout.
- Bounded preparation scanned 10,000 training images and retained 1,924 training-related crops: 1,589 training and 335 training-validation. Counts across those splits are green 500, red 500, unknown 500, yellow 424. The yellow shortfall is reported, not filled from evaluation data.
- GT color evaluation has 1,389 development and 1,240 holdout usable crops. Detector-aware evaluation retains all 1,246 holdout GT light records, including six tiny/otherwise excluded GT crops and detector misses.
- Historical human residual labels were used by old training and are excluded from independent risk evaluation. No risk-accuracy improvement is claimed.
- Independent verification checked the prepared file hashes and cross-split group/pixel identities. This does not authenticate the original annotation archive. See [data preparation](EVALUATION_DATA.md).

## Detection comparison

Both models used CPU, two Torch threads, input size 640, NMS IoU 0.45 and maximum 300 detections. AP uses confidence 0.001 and 101-point interpolation; operational recall uses confidence 0.25 and IoU 0.5. These metrics omit COCO area/crowd handling and must not be called official COCO benchmark results.

| Holdout metric | YOLO11n baseline | YOLO11s candidate |
| --- | ---: | ---: |
| mAP50–95 | 13.92% | 19.33% |
| mAP50 | 23.55% | 32.54% |
| Person recall | 24.74% | 36.24% |
| Car recall | 37.15% | 43.95% |
| Traffic-light recall | 6.34% | 15.57% |
| Observed process RSS | 475.77 MiB | 565.35 MiB |

The mAP50–95 gain is 5.40 percentage points. Rider and traffic-sign classes are unsupported by this comparison and excluded explicitly. The train class has only one holdout instance; per-class support must accompany aggregate metrics.

The original n-then-s run recorded holdout P95 pipeline latency of 171.73 ms and 106.54 ms, respectively. A separate dependency installation overlapped part of that run; whole-run medians were 61.39 ms and 89.97 ms. This inconsistent tail ordering is not evidence that the larger detector is faster.

We therefore repeated actual pipeline timing on the same fixed 50 holdout images, reversing model order, with five warmups per round and three rounds, after other experiment/test work stopped:

| Repeated P95 pipeline latency | YOLO11n | YOLO11s |
| --- | ---: | ---: |
| Round 1 | 59.37 ms | 134.01 ms |
| Round 2 | 80.46 ms | 137.34 ms |
| Round 3 | 82.01 ms | 168.08 ms |
| Median of round P95 values | 80.46 ms | 137.34 ms |

The repeated P95 ratio is **1.707**, exceeding the 1.5 limit. These are single-workstation measurements, not universal hardware performance guarantees. The repeated run recorded no caught pipeline-error responses. The initial AP run recorded zero thrown processing exceptions but did not retain pipeline reason strings; the evaluation tool now records those and counts caught processing failures separately from ordinary abstention.

## Traffic-light color comparison

Both CNNs used frozen ImageNet-pretrained backbones, four classes, identical 128×128 preprocessing and five epochs. Checkpoints were selected only on training-validation loss. Values below apply the runtime confidence rule: confidence below 0.5 becomes `unknown`.

| Holdout GT-crop metric | ResNet18 baseline | MobileNetV3-Small candidate |
| --- | ---: | ---: |
| Macro-F1 | 67.27% | 66.99% |
| Green recall | 85.26% | 82.04% |
| Red recall | 68.57% | 72.14% |
| Unknown recall | 88.78% | 89.26% |
| Yellow recall | 50.00% | 50.00% |
| Crop P95 latency | 22.48 ms | 9.27 ms |
| Training duration | 113.80 s | 17.27 s |

The candidate loses 0.28 F1 points and 3.21 green-recall points. Holdout support is green 529, red 280, unknown 419, yellow 12; yellow is below the 20-instance support gate. Both runs finished within their 30-minute training limits. There were no crop-loading/classification failures.

Detector-aware color metrics include detector misses as false negatives in the GT denominator:

| Detector | Matched / 1,246 GT lights | Misses | ResNet18 macro-F1 | MobileNetV3-Small macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| YOLO11n | 79 | 1,167 | 8.34% | 8.10% |
| YOLO11s | 194 | 1,052 | 18.96% | 19.68% |

These are **GT-conditioned detector-aware classification** metrics. Unmatched detector false positives are measured in the detector precision/AP results; they are not silently claimed to be included in this color F1. The large gap from GT-crop metrics shows the detection bottleneck. See [CNN results](CNN_EXPERIMENT_RESULTS.md) for full policy and support details.

## Reproduction and artifacts

Use [detector evaluation](DETECTION_EVALUATION.md), [CNN experiments](CNN_EXPERIMENT.md), and the fixed manifests generated by [data preparation](EVALUATION_DATA.md). The local reports are `results/detection/report.json`, `results/detection/latency-repeat.json`, and `results/cnn/report.json`; model-specific JSONL/CSV files retain detailed predictions and hashes without publishing dataset images.

The measurements above were taken with Python 3.11.18, Torch 2.13.0+cpu, torchvision 0.28.0+cpu and Pillow 12.3.0, before the release security dependency update. The original environment is retained locally in `logs/experiment-environment-before-security.txt` and `results/cnn/environment.json`; the starting repository commit is recorded in [progress](REPRODUCTION_PROGRESS.md). The release uses patched dependencies and does not mislabel these historical measurements as measurements taken after that update.

Public model asset SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`. Candidate detector SHA-256: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`. The input image-manifest SHA-256 is `c2e506032b7b01d3271d7f722baf14fac32d0b6b7e2d248012c02f0950627a7c`.
