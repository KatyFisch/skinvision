# MILK10k MEL vs. NV dataset

From the repo root, run this after extracting the training release into `data/`.

```sh
uv run --locked python data_prep/create_splits.py
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
