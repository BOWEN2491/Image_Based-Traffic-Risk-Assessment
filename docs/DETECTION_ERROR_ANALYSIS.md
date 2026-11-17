# Bounded Detection Error Analysis

`tools/analyze_detection_errors.py` reads existing detector predictions and the
matching immutable `images.jsonl` manifest. It does not run inference, access
the raw dataset, or write its output into Git. Keep JSON summaries and case
identifiers under ignored `results/` paths.

```powershell
python tools/analyze_detection_errors.py `
  --predictions results/detection/yolo11n/predictions.jsonl `
  --manifest results/evaluation-bounded/images.jsonl `
  --output results/next-iteration/s5-analysis
```

When the locally configured source images are available, request an annotated
atlas explicitly and point at the dataset root:

```powershell
python tools/analyze_detection_errors.py `
  --predictions results/detection/yolo11n/predictions.jsonl `
  --manifest results/evaluation-bounded/images.jsonl `
  --image-root "<local dataset root>" --atlas --max-cases 100 `
  --output results/next-iteration/s5-analysis-atlas
```

The tool resolves each manifest image path under `--image-root`, rejects paths
that escape that root, and draws the associated false-negative/localization
box or unmatched prediction. The atlas and index are local only. Without
`--atlas` the tool writes metadata cases only and never opens source images.

The report separates development and holdout, records class precision at the
configured confidence threshold, and reports GT support, recall, and
size-assigned precision for small (`area/image < 0.001`), medium (`0.001` to
`<0.01`), and large (`>=0.01`) boxes. An unmatched prediction with positive IoU
is assigned to the size and class of its closest overlapping GT for that size
precision denominator; unmatched predictions with no overlap are included in
overall class precision but not attributed to an object-size stratum. Per-GT
failures distinguish misses, low IoU, wrong class, and processing errors.
Occluded/truncated breakdowns use manifest attributes where present and report
`unknown` otherwise.

The optional `cases.jsonl` is capped at 100 records by default and remains local.
It contains identifiers, box error summaries, and processing error strings,
not rendered images. This makes it suitable for selecting a later manually
reviewed image atlas without automatically publishing road images. The tool
cannot establish label correctness or source provenance. The historical
`evaluation-bounded` manifest predates BDD context/occlusion metadata, so those
attributes correctly appear as unknown in this analysis.

For the existing YOLO11n artifacts, the bounded rerun analyzed 502 development
and 498 historical holdout frames. At confidence 0.25 and IoU 0.5, traffic-light
recall was 36/1,402 (2.57%) on development and 79/1,246 (6.34%) on holdout.
Small-light recall was 2.93% and 2.78%, respectively; medium-light recall was
34.68% and 29.17%. This is an exploratory analysis of the existing historical
predictions and unverified annotations, not an independent test or an upgrade
claim. Machine-readable output is stored locally at
`results/next-iteration/s5-analysis-verified/`.
