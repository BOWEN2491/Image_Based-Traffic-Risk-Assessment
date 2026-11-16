# Bounded CPU traffic-light experiment

Prepare the source-grouped crop manifest first using [EVALUATION_DATA.md](EVALUATION_DATA.md). The recorded run used the bounded preparation output below (the earlier interrupted `results/evaluation` directory is not an experiment input):

```powershell
$env:TORCH_HOME = "$PWD/.cache/torch"
python -m traffic_risk.cnn_experiment --manifest results/evaluation-bounded/crops.jsonl --output results/cnn
```

The experiment compares ResNet18 and MobileNetV3-Small with ImageNet pretrained
backbones, 128x128 RGB input, four fixed classes, seed 42, and two CPU threads.
Only the final classification layer is trained; backbone weights and batch
normalization statistics stay frozen. AdamW uses learning rate 0.001, batch size
32, at most five epochs and a 30-minute budget per architecture. The budget is
checked between batches; a single batch cannot be interrupted. Pretrained
weights download into TORCH_HOME on first use. No external training service is used.

The manifest must contain train, train-validation, development, and holdout
splits. Training plus train-validation may contain at most 2,000 crops. Group and
pixel identities must not cross splits, and every crop must match its recorded
SHA-256. Training-validation loss selects the saved checkpoint; holdout does not
affect training or checkpoint selection. The saved checkpoint is reloaded before
evaluation.

Each architecture emits weights, class indices, runtime-compatible CNN metadata,
and metrics in the output directory. The combined report includes confusion
matrices, class support/recall, macro-F1, failure rate, per-crop P95 elapsed time
(decode, preprocessing and inference), and process RSS at evaluation completion.
RSS is a snapshot, not measured peak memory. Main metrics apply the runtime
confidence policy: probabilities below 0.5 become unknown. Raw argmax metrics
are reported separately. Failed samples remain in class-support denominators.
Per-sample predictions are local outputs and must not be committed.

Fewer than 20 holdout samples in any class marks support insufficient. Reports
always prohibit promotion while annotation provenance is local-unverified.
Crop latency is not the complete detector-to-risk latency and cannot alone
satisfy a deployment speed gate. Ground-truth crop scores also do not measure
detector misses; detector-matched crops require a separate evaluation.

These are traffic-light classification metrics, not traffic-risk accuracy.
