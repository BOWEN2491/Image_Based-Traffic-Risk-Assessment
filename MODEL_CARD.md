# Model Card: Image-Based Traffic Risk Assessment v0.1.0

## Summary

This research demo combines three components: an Ultralytics YOLO detector, a CNN that classifies detected traffic-light crops, and an XGBoost classifier operating on engineered scene features. Deterministic rules can override or reject a model estimate.

The output categories are `low`, `medium`, `high`, and `unknown`. They are qualitative demo labels, not calibrated probabilities or validated measures of road danger.

## Intended use

- Education and exploration of multi-stage computer-vision pipelines.
- Reproducibility experiments on user-supplied road-scene images.
- Research into feature engineering, weak supervision, and explicit abstention.

## Out-of-scope use

Do not use the models for vehicle control, route selection, driver alerts, insurance, law enforcement, emergency response, worker monitoring, or other decisions that affect safety or rights. Do not treat a `low` result as evidence that a scene is safe.

## Training data and labels

Development used BDD100K-associated road imagery obtained outside this repository. The repository does not distribute those images or annotations. Dataset composition, collection geography, camera placement, weather, lighting, and annotation policy can all introduce bias.

Risk targets were primarily weak labels generated from engineered features, clustering, and heuristic rules. The classifier therefore learns agreement with that labelling process. Similar features are used both to construct labels and to train the classifier, which can inflate apparent validation performance.

## Evaluation

Any historical accuracy near 96% must be described only as **agreement with weak labels on the evaluated split**. It is not a real-world traffic-risk accuracy claim. It does not establish calibration, collision prediction, causal validity, or safe operation.

A trustworthy future evaluation should use independently labelled scenes, source- or scene-grouped splits, per-class precision/recall/F1, confusion matrices, rule coverage, abstention rate, and failure rate. Results should also be stratified by lighting, weather, geography, camera type, and vulnerable-road-user presence.

## Known limitations

- A single image lacks motion, intent, depth, road friction, map context, and driver-state information.
- Detector errors propagate into every downstream stage.
- Small, occluded, distant, unusual, or overexposed traffic lights may be missed or misclassified.
- Domain shift can occur across countries, signage, camera lenses, seasons, weather, and night scenes.
- Engineered rules encode subjective assumptions and may fail in uncommon road layouts.
- Weak supervision can reproduce its own heuristics while appearing highly accurate.
- The model has not been independently validated for demographic or geographic fairness.
- Explanations describe the pipeline's inputs and rules; they are not causal explanations.

## Failure and abstention behaviour

Missing models, schema mismatch, corrupt images, unusable detections, non-finite features, or invalid classifier output must produce `status="uncertain"` with `risk="unknown"`, or a structured API error. The system must not convert a perception failure into a normal low-risk result.

## Model formats and integrity

- YOLO: upstream Ultralytics weights loaded through the Ultralytics runtime.
- Traffic-light CNN: weights-only state dictionary, loaded strictly against the expected architecture.
- Risk classifier: XGBoost native JSON/UBJ, not Python pickle/joblib deserialization.

The versioned release manifest is the source of truth for filenames, origins, sizes, SHA-256 digests, class mapping, and feature-schema compatibility. The provided downloader verifies assets before use.

## License

Repository code is AGPL-3.0-only. Third-party code, upstream model assets, and datasets retain their own terms; see `THIRD_PARTY_NOTICES.md`.
