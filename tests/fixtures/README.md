# Test fixture provenance

Public CI must not add road images, videos, model weights, or private uploads to this directory without documented redistribution rights. At present, no real road-image fixture is included. `tests/test_public_rules_detection.py` uses controlled in-memory detector results; it is a synthetic behavior regression, not evidence of real-road detection quality.

Before adding a distributable image fixture, add one entry per file with all of the following verified fields:

```text
- File:
- Author / data provider:
- Original source URL or package identifier:
- License and a direct link to its terms:
- Permission covering storage in this repository and public CI use:
- Download date and any conversion/cropping steps:
- SHA-256 of the committed bytes:
- Expected invariant (for example, a supported category is present):
- Reviewer and review date for the provenance record:
```

Do not infer permission from a public URL, familiar dataset name, or an existing local copy. If any field cannot be independently confirmed, keep the image local and record the public fixture gap in the release notes. Avoid assertions tied to exact floating point boxes or overly narrow confidence values.
