# S5 YOLO11n CPU Parameter Exploration

This experiment compares operational confidence and inference size using the
same local public YOLO11n checkpoint. It is a development-only exploration.
The source manifest remains `local-unverified`, there is no new independent
holdout, and no candidate can be promoted.

The runner deterministically selects one image per development ride group,
prioritizing 150 groups with a small/tiny traffic-light object and 50 groups
with no traffic light. If a stratum is too small, it fills from unused groups
and records the realized counts. It caps selection at 200 frames, hashes every
selected image before inference, and hashes the input manifest, config,
checkpoint, selected image set and runner. It writes detailed predictions only
to the local ignored output directory.

```powershell
python tools/run_s5_cpu_experiment.py `
  --config local/detection.json `
  --model models/yolo/yolo11n.pt `
  --output results/next-iteration/s5-cpu-experiment
```

The eight serial configurations are B0 640/.25, C1 960/.25, C2 640/.15, C3
960/.15, C4 full frame plus top-60% traffic-light ROI at both 640 and 960 with
.25 confidence, and C5 the same ROI resolutions with .15 confidence. Every inference pass uses CPU, two Torch threads,
confidence .001 for AP, IoU .45 NMS, and at most 300 detections. Operational
metrics filter at the named confidence and match at IoU .5. ROI coordinates are
translated back to the original image (the crop is top-left anchored) and
traffic-light boxes from both passes are merged with same-class IoU .45 NMS;
other categories come only from full-frame inference. The ROI timings include
both model calls and the merge.

Each configuration stops at 15 minutes and the overall experiment at 90
minutes. Per-configuration budgets are capped by the remaining total time.
The report retains partial results and marks timeouts; a partial
configuration is not ranked as complete. Per-image timing includes inference
and ROI merge, with P50/P95 and peak observed process RSS. AP is the project's
101-point implementation and does not implement COCO area/crowd behavior.
This exploration uses no training and does not access holdout rows.

## Completed development-only run

Run window: 2026-11-16 20:00:10–20:03:04 America/New_York (173.48 s total).
The checked-out code revision was `e062927854005da7f245b53609d93b31cb8ed61d`;
runner SHA-256 is recorded in the local summary. Environment: Windows, Intel
Core i7-13620H (13th generation), 16 GiB RAM, Python 3.11.18, Torch
2.13.0+cpu, Ultralytics 8.4.126, Pillow 13.0.0, CPU with two Torch threads.
All eight configurations completed 200/200 frames with zero processing
failures and no timeout. Per-config timing and RSS refer to the detector path
(image decode + YOLO inference + ROI merge), not the complete
`run_single_image` feature/rule pipeline.

The same fixed development selection was used for all configurations: 200
distinct ride groups, 150 frames with a small/tiny traffic light, and 50 frames
without traffic lights. This intentionally stress-focused composition is not a
natural-distribution sample. Traffic-light support in these 200 frames was 749
instances. The 200-frame fixed selection was hashed and saved locally.

| Configuration | mAP50–95 | Δ vs B0 | TL recall | Δ vs B0, paired group-bootstrap 95% CI | TL precision | Detector P95 ms | P95 ratio vs B0 | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0, 640/.25 | 14.17% | — | 7.08% | — | 50.96% | 46.09 | 1.00 | 410.7 |
| C1, 960/.25 | 15.86% | +1.69 pp | 13.48% | +6.41 pp [4.02, 8.55] | 56.74% | 76.15 | 1.65 | 462.5 |
| C2, 640/.15 | 14.17% | 0.00 pp | 10.28% | +3.20 pp [2.17, 4.60] | 39.29% | 50.71 | 1.10 | 469.7 |
| C3, 960/.15 | 15.86% | +1.69 pp | 20.29% | +13.22 pp [10.32, 15.90] | 44.97% | 76.23 | 1.65 | 469.7 |
| C4-640, ROI/.25 | 14.17% | −0.01 pp | 7.88% | +0.80 pp [0.00, 1.67] | 49.17% | 95.17 | 2.07 | 473.8 |
| C4-960, ROI/.25 | 15.88% | +1.71 pp | 17.22% | +10.15 pp [7.21, 12.64] | 56.58% | 158.56 | 3.44 | 498.8 |
| C5-640, ROI/.15 | 14.17% | −0.01 pp | 11.62% | +4.54 pp [3.26, 5.94] | 37.18% | 97.26 | 2.11 | 488.8 |
| C5-960, ROI/.15 | 15.88% | +1.71 pp | 23.36% | +16.29 pp [12.93, 18.93] | 42.17% | 167.75 | 3.64 | 504.7 |

Key operational class metrics on the same sample (recall/precision at the
configuration threshold, percentages):

| Configuration | Person R/P | Car R/P | Traffic-light AP50 / AP50–95 |
| --- | ---: | ---: | ---: |
| B0 | 25.93 / 76.24 | 38.75 / 84.67 | 10.92 / 3.74 |
| C1 | 35.02 / 79.39 | 44.40 / 85.88 | 20.87 / 7.17 |
| C2 | 33.33 / 67.81 | 43.93 / 73.99 | 10.92 / 3.74 |
| C3 | 43.77 / 63.41 | 50.61 / 74.39 | 20.87 / 7.17 |
| C4-640 | 25.93 / 76.24 | 38.75 / 84.67 | 11.06 / 3.69 |
| C4-960 | 35.02 / 79.39 | 44.40 / 85.88 | 22.47 / 7.32 |
| C5-640 | 33.33 / 67.81 | 43.93 / 73.99 | 11.06 / 3.69 |
| C5-960 | 43.77 / 63.41 | 50.61 / 74.39 | 22.47 / 7.32 |

Intervals use 200 paired bootstrap draws of the 200 ride groups with seed 42;
the same sampled group IDs were applied to each candidate and B0. The
bootstrap script and tests are included. To keep the resampling run under a
60-second CPU budget, it bootstraps traffic-light operational recall only;
mAP50–95 differences are point estimates with no confidence interval. These
intervals quantify variability within this preselected development sample,
not independent-holdout uncertainty.

`results/next-iteration/s5-cpu-experiment/` contains ignored local
`summary.json`, per-config prediction JSONL, selection manifest,
`bootstrap-final.json`, and byte-identical `runner_used.py` and
`bootstrap_tool_used.py` snapshots. The inference runner SHA-256 is
`959a187ba6824aa1540b7ed46ac730a1bfbd22591255b28e0f4d0bcf101c8d90`; the
bootstrap tool SHA-256 is `7af0ffb4684dea4e02c28f2ae40b9501e306cc50b71bdace2d9d61dc3e599d0a`.
The mAP definition is the repository's 101-point AP at
confidence .001 and IoU thresholds .50:.05:.95, not official COCO metrics.
Operational recall/precision use the configuration's confidence and IoU .5.
Lowering the confidence did not change AP, as expected, while traffic-light
precision dropped in C2/C3 and C5 variants.

The P95 detector-path ratios exceed 1.5 for all 960 and ROI variants. C2 stays
below that ratio but has no mAP gain and lower traffic-light precision. These
timings exclude feature extraction and rule evaluation, so the plan's 1.5×
complete-pipeline gate remains **unverified**. Source provenance is still
local-unverified and there is no new independent holdout. Therefore
`promotion_allowed=false`; the shipped model and default configuration remain
unchanged. No real-world risk accuracy claim is made.
