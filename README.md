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

## Train and run a classifier

Open [Train and use a clinical-photo classifier](notebooks/02_train_classifiers.ipynb)
to configure the data path and task, prepare the photos, train, evaluate, and save
a model, then classify a curated image.

```sh
uv run --locked jupyter lab --LabApp.extension_manager=readonly notebooks/02_train_classifiers.ipynb
```

See the [classifier setup](classifiers/README.md) for checkpoint requirements and
the [experiment record](docs/classifier-experiments.md) for what we tried and why
we kept the padded-crop and direct 11-class recipes.

## Run the local photo demo

Place the U-Net checkpoint at `models/unet_segmentation_full.pth`, or keep the
extracted checkpoint in `models/unet_segmentation_full/`. Use
`--segmentation-model` to load it from another location.

```sh
uv run --locked python -m demo --model data/models/padded-crop
```

Point `data/models/padded-crop` at a saved padded-crop model directory containing
`heads.joblib` and `metrics.json`. The demo uses that path by default when it
exists. Pass the directory from the notebook's save step with `--model` to use
a newly trained model.

Open `http://127.0.0.1:8765` and choose a curated JPEG or PNG. Drag the photo
or use the arrow keys to move it within the preview. Adjust **Zoom** to change
the framing, then select **Analyze photo**. The U-Net finds the lesion area in
the 600 × 450 crop, and the preview traces the detected area. The classifier
uses the lesion-centered crop with surrounding skin and displays a melanoma
research score band. When the mask is empty, it uses the original crop and notes
that on the result. Processing stays in memory on your computer.

To run the saved legacy cascade, pass its `model.json` with `--model`. Keep the
checkpoints referenced by that file at their saved relative paths. The cascade
uses the black-background input and shows a melanoma score band on its
MEL-vs-NV branch.

To populate **Sample photos**, install the five public-domain or CC0 examples
with `uv run --locked python -m demo.samples` (requires `curl`). The 600 × 450 JPEGs
are stored under `data/demo-samples/` and can be used offline after setup. The
gallery includes the source and license for every photo, including two
melanoma examples from the National Cancer Institute. If you run the demo
with `MILK10K_DATA_DIR`, use the same setting when preparing the samples.

## Attribution

MILK study team. MILK10k. ISIC Archive, 2025, doi:10.34970/648456.

MILK study team. MILK10k Benchmark. ISIC Archive, 2025, doi:10.34970/262082.

Codella et al. Skin Lesion Analysis Toward Melanoma Detection, ISBI 2017.
[arXiv:1710.05006](https://arxiv.org/abs/1710.05006).
