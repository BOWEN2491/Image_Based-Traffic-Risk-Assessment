# Fixed local evaluation data

Run from an installed checkout using a local, Git-ignored JSON configuration:

```json
{"dataset_root": "/path/to/your/dataset"}
```

```bash
python -m traffic_risk.prepare_evaluation --config local/config.json --output results/evaluation-bounded
```

The installed command `traffic-risk prepare-evaluation` accepts the same arguments.
`traffic-risk evaluate-detection --help` and `traffic-risk cnn-experiment --help`
describe the separate downstream evaluation and bounded training commands.

The source root contains `bdd100_images_100k/100k/{train,val}`, matching
`bdd100k_labels/100k/{train,val}` JSON files, and optional
`bdd100k_info/100k/{train,val}` ride metadata. Source data is never modified.
An existing output directory is refused to protect previous experiments.

The defaults use seed 42, at most 1,000 validation images, and at most 2,000
training crops (500 per color, including the training-validation subset).
`--max-images` and `--max-crops` can reduce these limits, not exceed them.
Training scans stop at `--max-train-images 10000`; total preparation also has a
`--max-seconds 1800` limit. Either limit saves the samples already selected,
recording the stop reason and class shortfalls rather than inventing balance.
Time is checked between samples; the current file operation is allowed to finish.
Progress is printed every 250 examined paths. The JSON summary file is the
machine-readable final result; console output also contains progress messages.
Selected images are fully decoded; annotation names and box bounds are checked.
Invalid samples are excluded with explicit reasons, not silently corrected.

Outputs:

- `images.jsonl`: image/annotation paths relative to the configured dataset root,
  byte and pixel hashes, image dimensions, source split, group, experiment split,
  and verified object boxes/colors. Detection evaluation selects `source_split=val`.
- `crops.jsonl`: PNG paths relative to the output directory, class, source image
  and object IDs, crop box, group, split, crop hashes and parent hashes.
- `excluded.jsonl`: invalid samples, duplicate pixels and group collisions.
- `summary.json`: counts, seed, manifest hashes and explicit promotion blockers.

Group assignment uses `rideID` when available, with image-ID fallback recorded.
Validation groups are deterministically assigned to `development` or `holdout`;
train groups to `train` or `train-validation` (approximately 80/20). Validation
groups and image pixels are reserved before training selection. Duplicate image
pixels and crop pixels are excluded globally, even when filenames differ.
All crops from a source group stay in the same experiment split.

Only original-image annotation boxes supply color labels. BDD `none` maps to
`unknown`; crops smaller than six pixels in either dimension are omitted.
The four classes are green, red, unknown and yellow. Validation crops are emitted
separately and do not count toward the 2,000 training-crop limit. Detector-matched
crop evaluation must separately report missed detections; accuracy on ground-truth
crops alone is not end-to-end performance.

These local annotations lack an authenticated upstream archive manifest. Every
record therefore has `provenance=local-unverified` and `promotion_allowed=false`.
Results are exploratory and cannot promote a new default model. Historical
residual risk annotations are deliberately excluded because they were used in
training. This tool validates selected samples, not the entire 100,000-image dataset.

## Recorded local preparation (2026-11-14)

The bounded run in ignored `results/evaluation-bounded` completed after examining
1,003 validation paths and 10,000 training paths. It retained 1,000 validation
images (502 development / 498 holdout) and 621 parent training images.

- Training crops: 1,924 total, including 1,589 train and 335 train-validation;
  green/red/unknown each have 500, yellow has 424. The scan limit stopped selection;
  the missing 76 yellow samples were not fabricated or oversampled into the manifest.
- Ground-truth evaluation crops: 1,389 development and 1,240 holdout. The holdout
  contains only 12 yellow crops, below the 20-per-class support gate.
- All 1,556 selected source groups use rideID. No group spans experiment splits;
  selected image pixels and crop pixels are unique. All 4,553 saved crop SHA-256
  values were independently checked after preparation.
- Local `verification.json` records these checks; `summary.json` records counts,
  exclusions and manifest hashes. Source provenance remains unverified.

An earlier unbounded preparation was stopped and its incomplete crop directory
retained for audit only. It has no completed manifest and must not be used for
training or evaluation. All experiments use the bounded run above.
