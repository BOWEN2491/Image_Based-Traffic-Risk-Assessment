# CPU traffic-light comparison

Exploratory local experiment; public default models remain unchanged.

Both models used the same source-grouped 1,924 training/training-validation crops, fixed seed 42, 128x128 preprocessing and frozen pretrained backbones. Both selected epoch 5 by training-validation loss. Holdout consisted of 1,240 GT crops: green 529, red 280, unknown 419, yellow 12. Yellow support is below the 20-sample minimum. Local annotations have not been authenticated against an upstream archive.

| Model | Holdout macro-F1 | Green recall | Red recall | Unknown recall | Yellow recall | Crop P95 ms | Training seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18 | 0.6727 | 0.8526 | 0.6857 | 0.8878 | 0.5000 | 22.48 | 113.8 |
| mobilenetv3 | 0.6699 | 0.8204 | 0.7214 | 0.8926 | 0.5000 | 9.27 | 17.3 |

Main metrics apply the runtime policy: confidence below 0.5 becomes unknown. Raw argmax scores, confusion matrices, per-image predictions, RSS snapshots and validation histories remain in the ignored local report. Both classifiers completed all epochs and evaluated all GT crops without a decode/inference failure.

MobileNetV3-Small reduced measured crop P95 to 0.413 times the ResNet18 value, but macro-F1 decreased by 0.28 percentage points and green recall decreased by 3.21 percentage points. It fails the predefined quality improvement gate. These are per-crop timings, including decoding/preprocessing, not complete detector-to-risk latency.

**Decision: retain the baseline.** The quality gate fails; annotation provenance is unverified and yellow holdout support is insufficient. None of these results establish traffic-risk accuracy.

## Reproduction evidence

Crop manifest SHA-256: `94f858df46227ddfb2bfa714a879c09ee1b504048b2d8b5682d9f1141426087f`

Commands and configuration are documented in [CNN_EXPERIMENT.md](CNN_EXPERIMENT.md). Local detailed report: `results/cnn/report.json`; execution log: `logs/cnn-experiment.log`. Model binaries and source/crop images are excluded from Git.

| Checkpoint | SHA-256 |
|---|---|
| resnet18 | `54ace8793c19b94acfb98f82f99507e97a6ec2d6afa3e3bdc1a4276926cf3795` |
| mobilenetv3 | `1072eade55328dca458974e074e3d0ec36e8d64844a3b00be732957f38be9c55` |

## Detector-crop evaluation

Using the same holdout source images, match traffic-light predictions at confidence >=0.25 to annotated boxes at IoU >=0.5, one-to-one. Evaluate every labelled light; missed lights remain false negatives. Tiny predicted crops follow the runtime unknown policy. This population contains 1,246 lights (including six tiny GT boxes omitted from the 1,240 GT-crop experiment).

| Detector | Matched / GT lights | Detector misses | ResNet18 macro-F1 | MobileNetV3-Small macro-F1 |
|---|---:|---:|---:|---:|

| yolo11n | 79 / 1246 | 1167 | 0.0834 | 0.0810 |
| yolo11s | 194 / 1246 | 1052 | 0.1896 | 0.1968 |

These are **GT-conditioned detector-aware classification scores**, not complete
end-to-end detection F1: unmatched additional detector predictions are assessed
by the separate detection benchmark, not counted again here. Detector misses
account for every failure in these two classifier runs; there were no additional
crop decoding/classifier exceptions. Detector coverage, especially for small
traffic lights, dominates this experiment. Classification quality on ideal GT
crops cannot be presented as deployed pipeline quality.

Reproduce after the detection benchmark (repeat for `yolo11s`):

```bash
python -m traffic_risk.cnn_detected_eval --matches results/detection/yolo11n/matched_traffic_lights.jsonl --checkpoints results/cnn --output results/cnn/detected-yolo11n.json
```

Detailed confusion matrices, class recall, missing counts and per-crop
predictions are in the ignored detector-specific CNN reports. No model is
promoted from these exploratory results.

## Recorded environment

The reported training and scores were produced before the release dependency
security refresh, using Python 3.11.18, torch 2.13.0+cpu, torchvision 0.28.0+cpu,
Pillow 12.3.0, NumPy 2.4.6 and Ultralytics 8.4.126. The original CPU lock is
available at baseline commit `2edfadd0404846c60a055716e339bc02d8b15949`.
Local detailed records are `results/cnn/environment.json` and
`logs/experiment-environment-freeze.txt`. Dependency updates are separately
verified for compatibility; these existing benchmark figures must not be
presented as a fresh run on an updated environment.
