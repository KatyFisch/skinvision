# Frozen MILK10k split

The [training notebook](../../notebooks/02_train_classifiers.ipynb) creates or
verifies both manifests from the original MILK10k release. To generate or verify
just the manifests from the repository root, run the command below.

```sh
uv run --locked python -m classifiers.eval_split.create_splits --include-full
```

Pass `--data-dir /path/to/data` or set `MILK10K_DATA_DIR` to use shared data.
Both CSVs stay in the ignored `splits/` directory under the data directory.

- `milk10k-v1.csv` records 5,240 lesions across all 11 diagnoses, with 4,192
  development lesions and 1,048 holdout lesions.
- `milk10k-mel-nv-v1.csv` retains 957 development and 239 holdout lesions from
  the parent split and adds the binary target, with NV as 0 and MEL as 1.

Each row records clinical and dermoscopic paths relative to the data directory,
diagnosis, duplicate group, split, and development fold. The seed-42 generator
recreates the original split before filtering. Paired images and exact duplicates
stay together. The source lacks patient IDs. An existing version is verified
before use. Changed assignments require a new manifest version. Use the same
input release and lockfile to reproduce the manifests.

Calibration groups are reserved within the development folds. Each ensemble head
fits on the non-calibration groups from two folds. The holdout is evaluated
after calibration fixes the cutoff. [MILK10k training release](https://doi.org/10.34970/648456)
