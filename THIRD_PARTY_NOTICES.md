# Third-Party Notices

This file records the principal external components and data sources used by Image-Based Traffic Risk Assessment. It is not a substitute for reviewing the complete dependency graph or the terms shipped with each release asset.

## Ultralytics

The inference pipeline uses the `ultralytics` Python package and an Ultralytics YOLO model asset.

- Project: https://github.com/ultralytics/ultralytics
- License route used by this repository: GNU Affero General Public License v3.0
- License information: https://www.ultralytics.com/license

Ultralytics also offers a separate enterprise license. This repository does not grant or represent that commercial license. Redistribution and network use must comply with the applicable Ultralytics terms and the AGPL-3.0-only license of this project.

## BDD100K tooling and dataset

The project was developed using traffic-scene data associated with BDD100K.

- Toolkit repository: https://github.com/bdd100k/bdd100k
- Toolkit license: BSD 3-Clause
- Dataset home: https://www.vis.xyz/bdd100k/

The toolkit license and the dataset terms are distinct. No BDD100K image or annotation is distributed in this repository or its model release. Users must obtain data through the official source and determine whether their intended use complies with its terms.

## Python and JavaScript dependencies

Runtime and development dependencies are declared in `pyproject.toml` and `risk_frontend/package-lock.json`. Each package remains under its own license. Dependency declarations do not imply endorsement by their authors.

## Trained model assets

Release model files are distributed separately from the Git tree. The release manifest records each asset's source, format, size, SHA-256 digest, class mapping, feature-schema version, and training-code version where available. A model derived from third-party software or data remains subject to applicable upstream terms.
