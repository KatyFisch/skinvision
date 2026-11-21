# skinvision

A medical computer vision project.

## Set up

- Download the images, metadata, supplemental metadata, and ground truth from
  [ISIC's MILK10k downloads](https://challenge.isic-archive.com/data/#milk10k).
- Put the three CSVs and the `MILK10k_Training_Input` folder in `data/`.

## Explore the data

```sh
./scripts/notebook.sh
```

Run this from the repository root to open the
[data audit notebook](notebooks/01_data_audit.ipynb) in JupyterLab. `uv`
installs the project dependencies as needed. Press `Shift+Enter` to run each
cell and `Ctrl+C` in the terminal to stop JupyterLab. If the dataset lives in
another checkout, set `MILK10K_DATA_DIR` to its absolute path.

Clear notebook outputs before committing to keep lesion photos out of Git.

## Attribution

MILK study team. MILK10k. ISIC Archive, 2025, doi:10.34970/648456.

MILK study team. MILK10k Benchmark. ISIC Archive, 2025, doi:10.34970/262082.
