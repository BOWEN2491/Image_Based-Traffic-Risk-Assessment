## Summary

Describe the user-visible or research-facing change.

## Verification

- [ ] Python compile, Ruff, and pytest with 80% coverage pass.
- [ ] Frontend lint, tests, and production build pass when affected.
- [ ] New success, abstention, and failure paths have tests.

## Release and safety review

- [ ] No dataset, user image, model binary, generated output, secret, or personal path is included.
- [ ] API/schema/model compatibility and migration impact are documented.
- [ ] Weak-label agreement is not described as real-world traffic-risk accuracy.
- [ ] Perception/model failures still return `unknown` rather than low risk.
- [ ] Licensing and third-party notices are updated when dependencies or assets change.
