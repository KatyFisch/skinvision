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

Pull `main` and start the demo from the repository root:

```sh
uv run --locked python -m demo
```

The fitted MEL/NV and direct 11-class heads come with the checkout. On first
launch, the demo downloads missing router and U-Net checkpoints from the
`demo-checkpoints-v1` tag on `origin` using the same Git access as `git pull`.
If the checkout already has the tag, the demo uses that copy. It stores the
bundle under `data/checkpoints/` and reuses it on later runs.

The demo also reuses the DermFM encoder downloaded by the training notebook.
Keep `MILK10K_DATA_DIR` set to the same directory if you set it for the notebook.
If you still need the encoder, follow the [classifier setup](classifiers/README.md#set-up)
to enable the Hugging Face download on first launch.

To try a newly trained model, pass the directory from the notebook's save step
containing `heads.joblib` and `metrics.json`.

```sh
uv run --locked python -m demo --model data/models/padded-crop
```

Open `http://127.0.0.1:8765` and choose a curated JPEG or PNG. Drag the photo
or use the arrow keys to move it within the preview. Adjust **Zoom** to change
the framing, then select **Analyze photo**. The U-Net finds the lesion area in
the 600 × 450 crop, and the preview traces the detected area. The classifier
uses the lesion-centered crop with surrounding skin. When it returns MEL,
the card shows **Inconclusive**, **Moderate melanoma signal**, or
**Strong melanoma signal** according to the model's score band. Other
predicted diagnoses appear as possible matches. The browser console logs the
raw melanoma score or the score for the highest-ranked other diagnosis. The
pulsing Grad-CAM overlay highlights stronger positive contributions inside
the original U-Net outline.
It shows normalized heat above 0.30 and reaches at most 75% opacity. When the
mask is empty, the demo prompts the user to adjust the crop.
Processing stays in memory on your computer.

### Use different checkpoints

To use a saved ResNet cascade router and a different direct 11-class run:

```sh
uv run --locked python -m demo --router-model path/to/cascade/model.json --other-model path/to/direct-11-run
```

Pass `--router-model` to use a local router manifest and its saved cutoff.
Keep the manifest's relative paths to the fold checkpoints intact. The demo
also recognizes a local direct 11-class run at `data/models/direct-11/`.
`MILK10K_DATA_DIR` replaces `data/` when set. Use a trusted run containing
`metrics.json` and `heads.joblib`.
The demo also reuses an existing `models/unet_segmentation_full.pth` or its
extracted folder. Pass `--segmentation-model` for another location and
`--mel-nv-only` to run just the padded-crop classifier.

The default router scores the full frame at a cutoff selected to route 95%
of melanomas on the development folds. The MEL/NV branch uses
the padded crop and reports the predicted diagnosis and melanoma signal.
The other branch scores the full frame and shows the leading diagnosis among
the nine other classes. The combined pipeline has not been evaluated as a
clinical tool.

To run the saved legacy cascade, pass its `model.json` with `--model`. Keep the
checkpoints referenced by that file at their saved relative paths. The cascade
uses the black-background input and shows a melanoma outcome on its
MEL-vs-NV branch.

The seven 600 × 450 **Sample photos** are included in
[`demo/assets/demo-samples/`](demo/assets/demo-samples/) and available offline.
The gallery includes melanoma, nevi, basal cell carcinoma, and seborrheic
keratosis, with each photo's source, author, and license. To regenerate the
bundled crops from their verified sources, run
`uv run --locked python -m demo.samples --data-dir demo/assets` from the repository root.

## Attribution

MILK study team. MILK10k. ISIC Archive, 2025, doi:10.34970/648456.

MILK study team. MILK10k Benchmark. ISIC Archive, 2025, doi:10.34970/262082.

Codella et al. Skin Lesion Analysis Toward Melanoma Detection, ISBI 2017.
[arXiv:1710.05006](https://arxiv.org/abs/1710.05006).
