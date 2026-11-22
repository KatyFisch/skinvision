# Train the lesion classifiers

Use [Train and use a clinical-photo classifier](../notebooks/02_train_classifiers.ipynb)
to go from the raw MILK10k release to a saved model and a prediction.

## Set up

1. Put the extracted MILK10k clinical and dermoscopic images and the training
   CSVs in `data/` or set `MILK10K_DATA_DIR` to a shared dataset path.
2. Accept the [DermFM-Zero-Open research terms](https://huggingface.co/redlessone/DermFM-Zero).
   If the pinned checkpoint is absent from the local data directory, run
   `uv run --locked hf auth login` with Hugging Face access to the model.
3. For padded-crop training, use the U-Net checkpoint from the segmentation
   notebook. Set `SEGMENTATION_MODEL` to the exported `.pth` file. The direct
   11-class workflow uses the original clinical photos.
   [Segmentation notebook](../notebooks/segmentation.ipynb)
4. The final cell defaults to a bundled gallery sample. Set `SAMPLE_PHOTO`
   to try another curated JPEG or PNG.

Launch JupyterLab from the repository root.

```sh
uv run --locked jupyter lab --LabApp.extension_manager=readonly notebooks/02_train_classifiers.ipynb
```

Set `TASK` to `padded_crop` for MEL vs. NV or `direct_11` for all 11 MILK10k
diagnoses. The notebook also exposes the data and output paths, device, batch
size, calibration settings, and logistic regression regularization. Run it once
per task. Clear notebook outputs before committing.

## Data and outputs

The workflow creates or verifies the [frozen MILK10k splits](eval_split/README.md)
from the raw release. Set `CATALOG = Path("final_data.csv")` to use Katharina's
existing image catalog export. Catalog paths resolve relative to the dataset
directory. The loader verifies image IDs and labels against the frozen split
and records the catalog checksum. The classifier uses just the clinical photo.

The original clinical frames are 600 × 450. The direct model uses them at
native resolution. The padded-crop recipe frames the lesion with surrounding skin
and falls back to the original frame for an empty mask. Feature extraction
caches the frozen embeddings locally. The first run needs the checkpoint and
can take substantially longer than later runs. For curated inference, the
direct classifier limits the longest side to 600 pixels for larger images and
keeps the aspect ratio.

Each saved directory contains `heads.joblib`, `metrics.json`, and calibration
and holdout predictions. The notebook reloads the heads with the pinned
DermFM-Zero-Open weights. Padded-crop inference also needs the U-Net checkpoint.
Keep the raw data, training outputs, and downloaded encoder weights local.
Load model bundles from trusted sources.

The [demo](../README.md#run-the-local-photo-demo) uses the checked-in padded-crop
and direct 11-class heads and fetches the router and U-Net checkpoints from the
repository on first use. Pass a saved padded-crop directory with `--model` to
try a new run. The demo also accepts the existing ResNet cascade `model.json`
and its referenced checkpoints.

## Explain a padded-crop prediction

The [padded-crop Grad-CAM notebook](../notebooks/explain_padded_crop.ipynb)
loads the three fitted heads and MEL cutoff from the checked-in
`models/mel-vs-nv-padded-crop.pth`. It reuses the app's crop and requires the
local U-Net checkpoint and access to the
[pinned DermFM-Zero-Open weights](https://huggingface.co/redlessone/DermFM-Zero).
Each collaborator can download the encoder under their own Hugging Face
account.

```sh
uv run --locked jupyter lab --LabApp.extension_manager=readonly notebooks/explain_padded_crop.ipynb
```

To update the shared heads from a new padded-crop run, export the fitted heads
and cutoff from the repository root.

```sh
uv run --locked python -m classifiers.export_padded_crop data/models/padded-crop models/mel-vs-nv-padded-crop.pth
```

See the [experiment record](../docs/classifier-experiments.md) for the results,
selection rationale, and limits of the evaluation.
