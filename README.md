# skinvision

A medical computer vision project.

## Set up

- Download the images, metadata, supplemental metadata, and ground truth from
  [ISIC's MILK10k downloads](https://challenge.isic-archive.com/data/#milk10k).
- Put the three CSVs and the `MILK10k_Training_Input` folder in `data/`.

## Preprocess ISIC 2017 training images

1. Download **Training Data** and **Training Ground Truth** for Task 1 from the
   [ISIC 2017 Challenge downloads](https://challenge.isic-archive.com/data/#2017).
2. Extract the ZIPs into `data/` so the JPEGs are directly in
   `data/ISIC-2017_Training_Data/` and the PNG masks are directly in
   `data/ISIC-2017_Training_Part1_GroundTruth/`.
3. Run from the repository root.

   ```sh
   uv run --locked python preprocessing/resize_isic.py
   ```

The script center-crops and resizes both to 600 × 450 pixels. It writes images to
`data/ISIC-2017_Training_resizedImages/` and masks to
`data/ISIC-2017_Training_resizedSegmentation/`.

## Explore the data

```sh
./scripts/run_explore_notebook.sh
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

Codella et al. Skin Lesion Analysis Toward Melanoma Detection, ISBI 2017.
[arXiv:1710.05006](https://arxiv.org/abs/1710.05006).
