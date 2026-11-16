# Release 0.2.2 dependency verification

The existing main-branch Python audit failed before this work. The failure was reproduced locally: 61 advisories across six distributions. This update changes only those six Python pins and the project's Pillow/build-backend bounds; it does not suppress advisories.

| Distribution | Previous lock | Patched lock |
| --- | --- | --- |
| certifi | 2026.7.22 | 2026.10.18 |
| filelock | 3.32.3 | 4.2.10 |
| msgpack (development lock) | 1.2.1 | 1.2.3 |
| Pillow | 12.3.0 | 13.0.0 |
| setuptools | 84.0.0 | 84.1.0 |
| urllib3 | 2.7.0 | 2.8.0 |

Both CPU and development locks have matching versions wherever a package occurs. The project now requires Pillow >=13,<14; the build backend requires setuptools >=84.1.0. The targeted lock changes were validated with the complete resolver (`pip install -r requirements-dev.lock -e .`), rather than bypassing dependency resolution. Its installation report lists exactly the six packages above and this project. `pip check` reports no broken requirements.

Compatibility review:

- [Pillow 13 release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html): Python 3.10 is dropped, while this project targets 3.11/3.12. Removed mode-changing `fromarray` behavior, I-mode PNG saving and legacy helpers are not used by this project. RGB decoding/cropping/transforms and upload handling are covered by the post-update tests.
- [filelock changelog](https://py-filelock.readthedocs.io/en/latest/changelog.html): version 4 changes the SoftReadWriteLock disk protocol. This project does not use that protocol or share legacy locks; its dependencies permit filelock 4. The patched 4.2.10 release also includes subsequent lock correctness fixes.
- Frontend compatible dependency updates and the separately reviewed Happy DOM 21.0.4 change reduce npm audit findings from ten to zero. Node >=20 remains supported. The Happy DOM change disables external CSS/iframe fetching in its test DOM; the tests do not rely on those network loads.

The post-update Python audit reports no known vulnerabilities for audited distributions. It cannot audit the unpublished project distribution or the official `torch==2.13.0+cpu` and `torchvision==0.28.0+cpu` build identifiers, and reports those as skipped. This is not a claim that every binary has been audited.

Local evidence is retained in `logs/security-dependency-install.json`, `logs/pip-audit-fixed.json`, and `logs/npm-audit-final.json`. Experimental measurements were collected before the Python security update; see [experiment results](EXPERIMENT_RESULTS.md) for the recorded environment distinction.
