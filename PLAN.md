# SkinVision project plan

The goal is an educational app that explores possible skin lesion diagnoses
from an ordinary phone photo. The current classifier workflow uses MILK10k
clinical photos, with one MEL-vs-NV model for the local demo and one model
covering all 11 diagnosis codes.

## What we verified about MILK10k

The local training data contains **5,240 lesions and 10,480 JPEGs**. Every
lesion has one clinical close-up and one dermoscopic image. Dermoscopy uses a
specialized instrument. The clinical close-up is the closer match to the
intended input, and performance on clinic photos may still change with
self-taken photos, different devices, lighting, and framing.
[ISIC MILK10k dataset](https://api.isic-archive.com/doi/milk10k/)

- `MILK10k_Training_Input/` contains two JPEGs per lesion, stored as
  `<lesion_id>/<isic_id>.jpg`.
- `MILK10k_Training_Metadata.csv` has one row per image with modality, age band,
  sex, skin tone class, site, and image-derived MONET scores.
- `MILK10k_Training_GroundTruth.csv` has one row per lesion, joined by
  `lesion_id`. Each row marks exactly one of 11 diagnosis categories.
- `MILK10k_Training_Supplement.csv` has one row per image, joined by `isic_id`.
  It records the detailed diagnosis, how it was confirmed, and invasion
  thickness where available. Use it to audit labels.

The local image paths and CSV identifiers matched. Basal cell carcinoma (`BCC`)
accounts for 2,522 lesions, melanoma (`MEL`) for 450, and other malignant
proliferations (`MAL_OTH`) for nine. Always predicting `BCC` would achieve about
48% accuracy on the full training release. Nine examples are too few for a
precise class-specific performance estimate. Most labels were confirmed by
histopathology (5,016 lesions), with 224 based on clinical assessment. The
published dataset is licensed CC BY-NC.
[MILK10k dataset paper](https://pubmed.ncbi.nlm.nih.gov/40701400/)
[ISIC download and license information](https://challenge.isic-archive.com/data/#milk10k)

ISIC offers automated scoring on a blind holdout of 479 lesions from the same
sources. The download page lists ground truth as unavailable. We can use the
official scorer once the full pipeline can emit all 11 diagnosis scores.
Assess performance on self-taken phone photos in a separate evaluation.
[MILK10k benchmark and scoring](https://challenge.isic-archive.com/landing/milk10k/)

## 1. Audit the data in JupyterLab

Open the [data audit notebook](notebooks/01_data_audit.ipynb) with
`./scripts/run_explore_notebook.sh` and run it top to bottom. It addresses four
questions before we train a model.

1. **What is one example?** Build one row per lesion with both image paths, the
   diagnosis category, and metadata. Assert key uniqueness, complete joins, and
   one image of each modality. Check for unreadable files and record image
   dimensions and aspect ratios by modality.
2. **Who and what is represented?** Show lesion counts by diagnosis, skin tone,
   age, sex, and site. Report missing values and cross-tabulate diagnosis with
   confirmation method and skin tone, including the sample counts.
3. **What would a model see?** Review a small, reproducible gallery of paired
   images across common and rare classes. Look for framing, markers,
   instruments, and other cues that could let the model learn shortcuts.
4. **What could leak across a split?** Look for exact or near-duplicate images
   across lesion IDs. Look for patient or source identifiers. The local CSVs
   expose neither, so document what overlap can be checked and what remains
   unknown. Confirm which input fields would be available from a phone photo.

**Done when** both collaborators can run the notebook and agree on the unit of
analysis, label mapping, and known limits on splitting and evaluation.

## 2. Train and evaluate the retained classifiers

Run the [training notebook](notebooks/02_train_classifiers.ipynb) from the
raw MILK10k release. It uses the frozen split, prepares clinical close-ups,
fits the three-member ensemble, calibrates its melanoma cutoff on reserved
development groups, reports holdout performance, and saves and reloads the
model. Both tasks use a frozen DermFM-Zero-Open encoder.

- **Padded-crop MEL vs. NV.** Keep lesion context for the demo and fall back
  to the original frame when the U-Net returns an empty mask.
- **Direct 11-class.** Score all MILK10k diagnoses on the original clinical
  frame. Apply the calibrated melanoma cutoff and show the highest-scoring
  remaining diagnosis for photos below the cutoff in research evaluations.

Use the same [frozen assignments](classifiers/eval_split/README.md) for both
tasks. Keep paired images and exact duplicates together. Group by patient
when those identifiers are available. Report melanoma sensitivity and
specificity with counts and uncertainty. For the 11-class model, also report
macro F1, per-class recall with counts, and a confusion matrix.

The [experiment record](docs/classifier-experiments.md) covers the models
retired from training, their results, and why the current recipes were selected.

## 3. Test the phone-photo workflow with curated examples

- Keep the demo private and use curated, licensed examples. The app defaults
  to the router, padded-crop specialist, and direct 11-class model, with
  inference checkpoints downloaded on first launch. It can also load the
  historical cascade.
- Look for independently labeled, self-taken phone photos from the intended
  population. Check label compatibility and measure the effect of device,
  lighting, skin tone, image quality, and unfamiliar conditions. Test how often
  the model abstains and how often it makes errors among the remaining cases.
  Choose any abstention rule on development data and verify it on independent
  data, including out-of-scope photos.
- Treat public predictions on real user photos as a separate decision. Before
  pursuing that use, review the independent evidence with a dermatologist and
  address how people interpret the output, privacy for uploaded images, dataset
  permissions, and any applicable medical-device requirements.
  [IMDRF good machine learning practice principles](https://www.imdrf.org/documents/good-machine-learning-practice-medical-device-development-guiding-principles)
