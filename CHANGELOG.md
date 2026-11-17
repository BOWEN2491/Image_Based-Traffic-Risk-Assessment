# Changelog

## [0.3.0] - 2026-11-16

- Added auditable data provenance, cross-manifest leakage checks, and rare/small traffic-light support reporting. Source remains `local-unverified`, annotation review awaits a human, and no independent new holdout was established.
- Added bounded detector failure analysis and an exploratory CPU operating-point comparison. Candidate settings did not meet the full-pipeline latency gate; public detector and risk defaults remain unchanged.
- Added opt-in detection details from the same inference pass and accessible reason explanations / overlays in the frontend.
- No BDD-derived CNN/XGBoost weights, datasets, road images, or local experiment outputs are distributed. No claim of improved risk accuracy is made.

## [0.2.3] - 2026-11-16

- Added explicit inference reason codes so successful `unknown` abstentions are distinguishable from processing failures.
- Unified the standalone traffic-light CNN command, runtime, and experiment confidence policy across ResNet18 and MobileNetV3-Small.
- Added scheduled and change-triggered Windows/Linux CPU real-model smoke coverage while keeping controlled detector tests clearly identified as synthetic.
- Reused the public v0.2.1 YOLO asset bundle; no CNN/XGBoost weights or road images are added to the release.

## [0.2.2] - 2026-11-15

- Fixed public rules inference on traffic-light scenes without a local CNN and honored the configured inference device.
- Separated public rules smoke from explicitly configured local full-model verification.
- Added offline `doctor` diagnostics and reproducible workstation setup evidence.
- Unified local manifest generation, CNN architecture/preprocessing metadata, strict checkpoint loading, and native model round-trip tests.
- Added bounded source-grouped data preparation and CPU perception evaluation tools. Local annotation provenance remains unverified; no default detector or risk policy is upgraded by these experiments.
- Updated frontend dependencies, including a reviewed Happy DOM security update; retained the existing user interface and API contract.
- Patched six Python dependency pins, including reviewed Pillow 13 and filelock 4 compatibility, and validated the complete resolver and dependency audit without advisory exclusions.
- Retained the existing public YOLO asset bundle and its checksums. CNN/XGBoost weights and dataset-derived outputs remain local-only.

## [0.2.1] - Maintenance hardening

- Documented the rules-first public runtime and local-only CNN/XGBoost distribution boundary.
- Added separate runtime/API and whole-package coverage gates to CI.
- Added Python 3.12 to the cross-platform CI matrix.
- Updated API examples, model-card language, and model asset guidance for the v0.2.x contract.

## [0.1.1] - Source-only withdrawal notice

- Marked the v0.1.0 model manifest withdrawn; downloader and runtime fail closed.
- Fixed weak-label v2 danger fields and added deterministic, strict input validation.
- Documented the incomplete training provenance and unconfirmed BDD100K-derived weight redistribution basis.

All notable changes to this project are documented here. The format follows Keep a Changelog, and releases use Semantic Versioning where practical.

## [Unreleased]

## [0.1.0] - 2026-08-22

### Added

- Reproducible research-demo packaging, model manifest/downloader, and model card.
- Hardened FastAPI upload and readiness endpoints with explicit uncertainty responses.
- Cross-platform Python, frontend, dependency-audit, and secret-scan workflows.

### Changed

- Clarified that weak-label agreement is not real-world traffic-risk accuracy.
- Moved model binaries outside Git into checksum-verified release assets.

### Removed

- BDD100K images, generated results, training outputs, caches, local paths, and binary models from the published tree.

[Unreleased]: https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/releases/tag/v0.3.0
[0.2.3]: https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/releases/tag/v0.2.3
[0.2.2]: https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/compare/v0.2.1...3a8cfae60f329b13f95a1ce5fe1d00897d6f7b64
[0.2.1]: https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/releases/tag/v0.2.1
[0.1.0]: https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/releases/tag/v0.1.0

