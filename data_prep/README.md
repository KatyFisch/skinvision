# MILK10k evaluation splits

From the repo root, run this after extracting the training release into `data/`.

```sh
uv run --locked python data_prep/create_splits.py
```

Pass `--data-dir /path/to/data` or set `MILK10K_DATA_DIR` to use a shared
dataset from a worktree. The output is `data/splits/milk10k-v1.csv`, relative to
the chosen dataset directory. Keep the manifest with the local data.

The CSV assigns each lesion and its two image paths to `test` or `development`.
Paths are relative to the dataset directory. For a development run, pick one
`validation_fold` (1–3) for validation and train on the other two. Use the 20%
test holdout after choosing the model and preprocessing.

Version 1 uses the [MILK10k training release](https://doi.org/10.34970/648456)
and 24 grouped, diagnosis-stratified shuffles starting at seed 42. It checks
split size, then favors coverage across diagnosis, recorded sex, skin tone, age,
and anatomical site. It also favors coverage of diagnosis-by-sex and
diagnosis-by-skin-tone combinations with at least one expected case per fold.
Among equally covered splits, it selects the closest overall category counts and
prints them for review. Sparse intersections can still have few or zero
examples.

Matching JPEG bytes are grouped under the same `group_id`, including chains of
matches across both views. The source CSVs lack patient identifiers, and
re-encoded copies require a separate check. Rerunning with the same inputs and
lockfile reproduces the manifest. A changed assignment requires a new version.
