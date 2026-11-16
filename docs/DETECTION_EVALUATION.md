# CPU detector comparison

Prepare the immutable image/crop manifests with `traffic-risk prepare-evaluation` first. Keep the dataset, crop files, model checkpoints and reports outside Git. An image manifest carries the source image and label hashes, source split, group identity, and fixed development/holdout role.

Create an ignored `local/detection.json` configuration (paths are resolved from the working directory):

```json
{
  "dataset_root": "./data/bdd100k",
  "images_manifest": "results/evaluation-bounded/images.jsonl",
  "output_dir": "results/detection",
  "threads": 2,
  "models": {
    "yolo11n": "models/yolo/yolo11n.pt",
    "yolo11s": "models/candidates/yolo11s.pt"
  }
}
```

Run `traffic-risk evaluate-detection --config local/detection.json`. Model order is baseline, then candidate. The evaluator requires existing local weights and never silently downloads them. Candidate YOLO11s used for the initial experiment comes from [Ultralytics v8.3.0 assets](https://github.com/ultralytics/assets/releases/tag/v8.3.0); the report records its SHA-256. It is not automatically installed as the public default.

At most 1,000 `source_split=val` images are evaluated. Both models use CPU, the same thread count, 640 pixel inference, NMS IoU 0.45 and maximum 300 detections. AP predictions use confidence 0.001; operational recall uses confidence 0.25 and IoU 0.5. A separate actual rules-pipeline invocation is timed per image after warmup. Latency excludes loading/downloading the model; RSS is the observed whole-process footprint, not an isolated model allocation.

Outputs:

- `report.json`: development/holdout per-class AP, recall and precision, P50/P95 latency, observed RSS, failure rate, weight and input manifest hashes, and candidate deltas.
- `<model>/per_class.csv`: tabular per-class results.
- `<model>/predictions.jsonl`: every image's predictions and ground truth, including failures. Categories are mapped explicitly; unsupported rider and traffic-sign categories are excluded rather than silently called correct.
- `<model>/matched_traffic_lights.jsonl` and `crops/`: confidence >=0.25 light detections matched one-to-one at IoU >=0.5. Every ground-truth light has a row; a null crop denotes a miss. Crop paths are relative to that model's output directory. These support downstream color evaluation without hiding detector misses.

AP uses 101-point interpolated precision and IoUs 0.50 through 0.95. It does not implement COCO area/crowd handling and is **not an official COCO evaluation**. Classes without ground-truth support are reported as null and excluded from the macro average. Failed images retain their ground truth, so missed detections reduce recall.

For a bounded latency-only repeat, use `python -m tools.benchmark_latency --config local/detection.json --output results/detection/latency-repeat.json`. It reverses model order and runs three rounds over the same first 50 holdout images with five warmups per round. Stop other experiment/test workloads before timing. See [recorded results](EXPERIMENT_RESULTS.md) for the observed order/load sensitivity.

Local annotation provenance remains unverified. Reports therefore explicitly disallow promotion regardless of numerical improvement. A future independently verified run must also satisfy the agreed holdout thresholds: >=1 percentage-point mAP improvement, no >1 point drop in key-class recall, >=20 key-class instances, and P95 pipeline latency <=1.5 times the baseline. These perception measurements make no claim about real-world risk accuracy.
