# SkinVision

SkinVision explores possible skin lesion diagnoses from phone photos.\
The local demo shows classifier predictions on curated examples.

<https://github.com/user-attachments/assets/1a37cb33-6a45-4515-b219-e24a6088547f>

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

- MILK study team. [MILK10k](https://doi.org/10.34970/648456). ISIC Archive, 2025.
  MILK study team. [MILK10k Benchmark](https://doi.org/10.34970/262082).
  ISIC Archive, 2025. Both use
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
  The legacy Grad-CAM notebook adds model overlays to a dataset photo.
  [Dataset paper by Tschandl et al.](https://doi.org/10.1016/j.jid.2025.06.1594)
- We crop and resize images and masks from the
  [ISIC 2017 Challenge](https://challenge.isic-archive.com/data/#2017)
  ([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)).
  [Challenge paper by Codella et al.](https://arxiv.org/abs/1710.05006)
- Our classifiers use frozen DermFM-Zero-Open features. The
  [Hugging Face checkpoint terms](https://huggingface.co/redlessone/DermFM-Zero#intended-use-and-license)
  specify [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
  for noncommercial academic research and restrict redistribution of weights
  and derivative checkpoints. [DermFM-Zero paper](https://arxiv.org/abs/2602.10624),
  [PanDerm paper](https://doi.org/10.1038/s41591-025-03747-y).
- The demo and video use cropped and resized photos from the
  [sample catalog](demo/samples.py), including
  [Cherry angioma by JOAN](https://commons.wikimedia.org/wiki/File:Angioma_cereza_en_hombro.jpg)
  ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
  Diagnosis descriptions summarize Wikipedia articles and use the
  [MILK10k categories](https://challenge.isic-archive.com/landing/milk10k/).
