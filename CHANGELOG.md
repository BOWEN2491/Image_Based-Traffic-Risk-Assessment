# Changelog

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

[Unreleased]: https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BOWEN2491/Image_Based-Traffic-Risk-Assessment/releases/tag/v0.1.0

