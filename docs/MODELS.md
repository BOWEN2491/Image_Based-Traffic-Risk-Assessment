# Model assets

The v0.1.0 model release is withdrawn. Its tag and assets are retained for audit, but the packaged manifest deliberately refuses downloads and runtime loading. The withdrawal records a training/runtime contract mismatch, incomplete reproducibility, and an unconfirmed BDD100K-derived weight redistribution basis. The weak-label v2 fix is not asserted to have contaminated the v0.1.0 XGBoost because historical provenance is incomplete.

Model binaries are release assets, not Git objects. Use the repository downloader instead of copying files from an old checkout:

```bash
python -m tools.download_models
```

The manifest is installed as package data, so the same command works from an editable checkout or an installed wheel. The default destination is `./models` in the caller's current working directory; use `--destination` to select another location.

## Verification

The downloader must verify both the expected byte size and SHA-256 digest before moving a completed download into its final location. An existing file is reusable only if it passes the same checks. Partial or mismatched files must fail closed and must not be loaded.

The inference runtime independently reads the manifest packaged in the installed wheel and verifies the exact size and SHA-256 of all six assets before loading any checkpoint. It then requires the published class mapping, feature-order sidecar, metadata feature contract, XGBoost feature names/order, and schema version to match the code exactly. Any mismatch leaves `/ready` unavailable rather than attempting inference.

The versioned manifest records:

- release/tag and direct asset URL;
- filename, byte size, and SHA-256;
- upstream or training origin and applicable license;
- serialization format and loader constraints;
- feature-schema version for the XGBoost model;
- architecture/class mapping for the traffic-light CNN;
- package/training-code version where known.

For the v0.1.0 risk classifier, the historical training-time XGBoost version is unknown and is recorded as such rather than inferred. The release UBJ was converted and verified with Python 3.11.9 and XGBoost 3.2.0, and its supported loader range and exact 11-feature contract are recorded in the manifest.

## Safe formats

- Load the risk classifier from XGBoost native JSON or UBJ.
- Load the traffic-light CNN as a weights-only state dictionary and require exact architecture keys.
- Load YOLO weights only from the release manifest's declared Ultralytics source.

Never load an untrusted `.pkl`, `.pickle`, or `.joblib` file. Python deserialization formats can execute code during loading.

## Updating assets

A model update requires a new manifest version, digests calculated from the final uploaded bytes, a model-card update, compatibility tests, and a release note. Do not silently replace an asset at an existing release URL. Model tests that require large assets belong in the manual smoke workflow; ordinary pull-request tests use mocks.

