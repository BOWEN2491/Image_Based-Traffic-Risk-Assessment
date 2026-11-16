# Reproduction implementation progress

Baseline: `2edfadd0404846c60a055716e339bc02d8b15949` (public v0.2.1).
Working branch: `fix/reproducible-cpu`.

## Sequence and acceptance

1. P0: isolated Python 3.11 CPU environment; ignored legacy assets with SHA-256 provenance; offline doctor.
2. P1: rules mode traffic-light fix; separate public/full model smoke; CLI/API/browser real-model checks.
3. P2: shared model/manifest/training contracts and source-grouped evaluation records.
4. P3: bounded CPU YOLO11n/YOLO11s and ResNet18/MobileNetV3-Small comparison; preserve baseline unless promotion gates pass.
5. Release: fresh install, Windows/Linux 3.11/3.12 CI, frontend checks, documented real-model evidence.

## Research workflow references

- [Evaluation data and provenance](EVALUATION_DATA.md): bounded source-grouped preparation, independent integrity checks, and exclusion of historical training-used residual labels.
- [Detector evaluation](DETECTION_EVALUATION.md): exact AP/recall/timing definitions and detector-matched traffic-light crops.
- [CNN experiment](CNN_EXPERIMENT.md): bounded frozen-backbone comparison and runtime confidence policy.

Release target is 0.2.2, a reproducibility maintenance update. The public YOLO bundle remains v0.2.1; research tools and local-unverified experiments do not replace the public default model.

The bounded experiments are complete; see [aggregate results](EXPERIMENT_RESULTS.md). The candidate detector failed the repeated latency threshold and the CNN candidate failed F1/recall thresholds. Annotation provenance and rare-class support independently block promotion. See [dependency updates](DEPENDENCY_UPDATES.md) for the subsequent security patch and audit limitations.

## P0 evidence

- Original workstation directories remain read-only; copied artifacts are under ignored `local/legacy/`.
- Ignored `local/asset-provenance.json` records source, destination, bytes, SHA-256 for each copy.
- Ignored `local/config.json` contains workstation paths; no absolute workstation paths belong in Git.
- Environment is `.venv`; package/download caches are `.cache`; logs are `logs`.
- Runtime diagnostics: `traffic-risk doctor --config local/config.json` (or `python -m traffic_risk.doctor`).
- Doctor is offline and does not load checkpoints. Nonzero means installation/configuration/integrity checks failed; it does not claim inference was executed.
- Full dataset integrity, historical model provenance and independent risk labels remain unverified at this stage.

## Validation log

P0 executed on Windows, 2026-11-13:

- Python 3.11.18 isolated prefix created; `requirements-dev.lock` installed with `--no-deps`, then editable project with `--no-deps`.
- `python -m pip check`: no broken requirements.
- `pytest tests/test_doctor.py -o addopts='' -q`: 6 passed.
- Ruff on changed Python files: passed.
- `traffic-risk download-models`: downloaded public YOLO asset and verified size/SHA-256.
- `python -m traffic_risk.doctor --config local/config.json`: `ok: true`.
- Installation/download/doctor logs are kept under ignored `logs/`; this is installation/integrity evidence, not an end-to-end inference claim.
- Independent P0 verification passed: six tests, Ruff, pip check, three diagnostic entry-point exit codes, simulated missing dependency, eight asset SHA-256 records, no heavyweight import or directory creation during diagnostics. Linux and Python 3.12 execution remain CI work.

## Frontend and browser validation

- Initial npm audit: 10 findings. Compatible lock updates reduced this to one Happy DOM finding.
- Reviewed Happy DOM [v21 breaking changes](https://github.com/capricorn86/happy-dom/releases/tag/v21.0.0) and [21.0.4 fixes](https://github.com/capricorn86/happy-dom/releases/tag/v21.0.4). Node requirement remains >=20; the breaking change disables external CSS/iframe loads, which these tests do not use. Updated this test dependency explicitly to 21.0.4, without force-upgrading unrelated packages.
- Final npm audit: zero findings. ESLint, seven Vitest tests, and Vite production build passed on Node 22.19.0.
- Headless Chromium exercised a real image upload through Vite proxy to the real rules API, checked the rendered unknown response, simulated a busy/503 response, and rejected invalid file type. No browser JavaScript errors.
- Local-only browser evidence: `logs/browser/report.json`, `logs/browser/real-result.png`, `logs/browser-smoke.log`. Dataset images/screenshots are not committed.

## Final local release validation (2026-11-15)

- Patched CPU/dev locks: pip check passed; PyPI audit found no known vulnerabilities in audited packages. CPU Torch/Torchvision wheels are outside the PyPI audit coverage; see DEPENDENCY_UPDATES.md.
- Full Python suite: 137 passed, 83.96% coverage (80% gate).
- Runtime/API suite: 69 passed, 85.64% coverage (85% gate).
- Compileall, Ruff, frontend lint, seven frontend tests, production build and whitespace checks passed.
- Final 0.2.2 wheel installed into an isolated environment outside the source tree; packaged resources, CLI imports, dependency checks, fresh public model download/checksum, ready/API smoke and doctor passed.
- Real road-image API and browser upload checks passed. Local images, weights and logs remain ignored.
- Bounded experiments completed; see EXPERIMENT_RESULTS.md and CNN_EXPERIMENT_RESULTS.md. Both public default models remain unchanged. Repeated detector timing did not meet the candidate latency gate, and CNN quality did not improve.
- Windows/Linux Python 3.11/3.12 hosted CI remains pending until the working branch PR runs. No hosted verification or release publication is claimed by the local results above.
