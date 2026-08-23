# Contributing

Thank you for improving this research demo.

## Before opening a change

- Use an issue for substantial behavioural or model changes.
- Never commit datasets, user images, model binaries, generated predictions, credentials, local paths, or training logs.
- Keep claims proportional to evidence. Weak-label agreement must not be presented as real-world risk accuracy.
- Preserve the invariant that perception/model failures abstain with `risk="unknown"`.

## Development setup

Use Python 3.11 and Node.js 20.19+ (or 22.12+):

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
cd risk_frontend
npm ci
```

Large models are not required for ordinary tests. Tests should use mocks or small generated images.

## Required checks

From the repository root:

```bash
python -m compileall -q src tests tools
ruff check src tests tools
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

From `risk_frontend/`:

```bash
npm run lint
npm test
npm run build
```

Add tests for success, abstention, and failure paths. API changes must update README examples and frontend handling.

## Pull requests

Keep pull requests focused and describe the user-visible effect, test evidence, model/schema compatibility, and any security or licensing impact. By contributing, you agree that your contribution is licensed under AGPL-3.0-only.
