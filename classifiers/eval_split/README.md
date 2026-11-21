# Lesion classifier data preparation

## MILK10k split

From the repo root, run this after extracting the training release into `data/`.

```sh
uv run --locked python -m classifiers.eval_split.create_splits
```

Pass `--data-dir /path/to/data` or set `MILK10K_DATA_DIR` to use a shared
dataset from a worktree. The output is `splits/milk10k-mel-nv-v1.csv` inside the
dataset directory. It stays local with the dataset.

Each row keeps both image paths for a melanoma (`MEL`, `target=1`) or nevus
(`NV`, `target=0`). Paths are relative to the dataset directory. Use the CSV to
select images directly from the original download.

The manifest assigns 957 lesions to `train` and 239 to `test`. During training,
hold out one `validation_fold` (1–3) at a time for model and threshold
selection. Use `test` after fixing the model and threshold.

The generator recreates the original seed-42 split on the
[MILK10k training release](https://doi.org/10.34970/648456) before filtering, so
each retained lesion keeps its test assignment or validation fold. Exact image
copies share a `group_id`. The source lacks patient IDs. Rerun with the same
inputs and lockfile to reproduce the manifest. Changed assignments require a new
version.

## Routed MILK10k tasks

From the repository root, regenerate the image catalog and export the full
MILK10k split before preparing the routed tasks.

```sh
uv run --locked python preprocessing/preprocess.py
uv run --locked python -m classifiers.eval_split.create_splits --include-full
uv run --locked python -m classifiers.eval_split.prepare_routed_tasks
```

The full split is `splits/milk10k-v1.csv`. Its `train` label corresponds to
`development` in the archived parent export. The holdout and three folds
retain their original memberships. Preparation checks those assignments
against `milk10k-mel-nv-v1.csv` and validates labels and image IDs for both
views against the catalog. `--catalog final_data.csv` and `--data-dir` let you
point to a specific export or shared dataset directory.

The outputs are `splits/milk10k-router-v1.csv` (5,240 lesions) and
`splits/milk10k-other-diagnoses-v1.csv` (4,044 lesions). Each has one row per
lesion with both paths, the original diagnosis, `target_name`, numeric
`target`, duplicate group, split, and validation fold. The nine-class targets
are zero-based in diagnosis-code order. Keep these CSVs under ignored `data/`.
Use `split=train` for development and preserve `split=test` for evaluation.

## ISIC 2017 MEL vs. NV split

Use the resized JPEGs from the [ISIC 2017 setup](../../README.md#preprocess-isic-2017-training-images)
and `data/ISIC-2017_Training_Part3_GroundTruth.csv` from
[Task 3](https://challenge.isic-archive.com/landing/2017/44/).

```sh
uv run --locked python -m classifiers.eval_split.split_isic2017
```

Exclude seborrheic keratosis. Screen resized ISIC images against the MILK10k
dermoscopy images using perceptual hashes and pixel agreement on two central
thumbnail crops, checking rotations and reflections. Exclude suspected
cross-source matches and keep suspected repeats within ISIC together. The
screen can miss different photos of the same lesion. Patient IDs are unavailable.

The seed-42, diagnosis-stratified split assigns about 60% / 20% / 20% to train,
validation, and test. It subdivides the 2017 challenge's training release. Outputs
are `splits/isic2017-mel-nv-v1.csv` and `splits/isic2017-mel-nv-overlap-v1.csv`
under `data/`. Rerunning with the same inputs reproduces those files.
