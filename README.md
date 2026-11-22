# SkinVision

SkinVision explores possible skin lesion diagnoses from phone photos.\
The local demo shows classifier predictions on curated examples.

## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Sign in to
Hugging Face, accept the research terms on the
[DermFM-Zero model page](https://huggingface.co/redlessone/DermFM-Zero), and log in
with a read token from that account so the pretrained encoder can download. Run
these commands from the repository root.

```sh
uv run --locked hf auth login
uv run --locked python -m demo
```

Open [the local demo](http://127.0.0.1:8765) and pick a sample photo. On the first
run, the demo also fetches the router and U-Net checkpoints using your Git access
to this repository.

[Watch the demo (MP4)](docs/skinvision-demo.mp4) to see the app analyze
**Melanoma close-up** and **Cherry angioma**.

For training, download the images, metadata, supplemental metadata, and ground
truth from [ISIC’s MILK10k downloads](https://challenge.isic-archive.com/data/#milk10k).
Put the extracted `MILK10k_Training_Input/` folder and all three CSVs directly
in `data/`.

To retrain the U-Net, also get the Task 1 training images and masks and the
Task 3 training labels from [ISIC 2017](https://challenge.isic-archive.com/data/#2017).
Extract them into `data/` and run both preprocessing scripts from the repo root.

```sh
uv run --locked python preprocessing/resize_isic.py
uv run --locked python preprocessing/preprocess.py
```

## Notebooks and training

- [Data audit](notebooks/01_data_audit.ipynb) checks labels, paired photos, and duplicates.
- [Segmentation](notebooks/segmentation.ipynb) trains the U-Net on ISIC 2017.
- [Padded-crop Grad-CAM](notebooks/explain_padded_crop.ipynb) visualizes what
  influenced a prediction. [Legacy Grad-CAM](notebooks/explain_classification.ipynb)
  explains the earlier classifier.

Open the [classifier training notebook](notebooks/02_train_classifiers.ipynb).

```sh
uv run --locked jupyter lab --LabApp.extension_manager=readonly notebooks/02_train_classifiers.ipynb
```

Set `TASK` to `padded_crop` for melanoma vs. nevus or `direct_11` for all 11
MILK10k diagnoses. For `padded_crop`, set `SEGMENTATION_MODEL` to
`repo_root / "data/checkpoints/demo-checkpoints-v1/unet_segmentation_full.pth"`
to reuse the demo’s U-Net.
Run the cells in order to train, evaluate, and save the model.

## Attribution

- MILK study team. [MILK10k](https://doi.org/10.34970/648456) and
  [MILK10k Benchmark](https://doi.org/10.34970/262082). ISIC Archive, 2025.
- Codella et al. [ISIC 2017 Challenge](https://arxiv.org/abs/1710.05006).
- Yan et al. [DermFM-Zero](https://arxiv.org/abs/2602.10624) and
  [PanDerm](https://doi.org/10.1038/s41591-025-03747-y).
- The demo and video use cropped photos with sources, authors, and licenses
  listed in the [sample catalog](demo/samples.py). Diagnosis descriptions
  summarize Wikipedia articles and use the
  [MILK10k categories](https://challenge.isic-archive.com/landing/milk10k/).
