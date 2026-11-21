# skinvision
a medical computer vision project


## Set up
* Download Training Data and Training Ground Truth from https://challenge.isic-archive.com/data/#milk10k
* Extract to data/ such that data/ contains the 3 .csv files and the MILK10k_Training_Input folder
* `uv sync` to install requirements

## Run the notebook

```sh
uv run notebook
```

Run this from the repository root to open `notebooks/segmentation.ipynb` in
JupyterLab in your browser. Press `Shift+Enter` to run a cell and `Ctrl+C` in
the terminal to stop JupyterLab.

## Attribution:
MILK study team. MILK10k. ISIC Archive, 2025, doi:10.34970/648456.

MILK study team. MILK10k Benchmark. ISIC Archive, 2025, doi:10.34970/262082.
