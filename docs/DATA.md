# Data preparation

## BDD100K

No BDD100K data is included in this repository. Obtain it from the [official BDD100K site](https://www.vis.xyz/bdd100k/) and review the dataset's current terms before use. Do not assume that the BSD-3-Clause license of the BDD100K tooling repository applies to the dataset itself.

Keep downloaded images and annotations outside the repository, or under the ignored local `data/` directory. Pass paths through the supported CLI/configuration interface; never add personal absolute paths to source files.

## Reproducibility record

For a training run, record without committing restricted data:

- dataset release and the official source used;
- a stable local split manifest or content hashes;
- source/scene grouping strategy;
- weak-label implementation version and random seed;
- feature-schema version and class mapping;
- package lock or environment export;
- per-class metrics, confusion matrix, rule coverage, and abstention/failure rates.

Do not publish a trained asset until its redistribution rights and upstream model terms have been reviewed.

## Test data

Automated tests must generate tiny synthetic images in temporary directories. Test fixtures must not be cropped from BDD100K or another restricted dataset.
