# SkinVision project plan

The goal is to learn how to build and evaluate a classifier that can suggest
possible skin lesion diagnoses from a single ordinary phone photo. Start with a
reproducible data audit, then test how far a simple classifier gets. Use the
results to decide what a private educational demo can responsibly show.

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

ISIC offers automated scoring today on a blind holdout of 479 lesions from the
same sources. The download page lists ground truth as unavailable. We can use
the official scorer once we freeze a model. For development, set aside labeled
validation data from the training release. Assess performance on self-taken
phone photos in a separate evaluation before making any claims about that use.
[MILK10k benchmark and scoring](https://challenge.isic-archive.com/landing/milk10k/)

## 1. Audit the data in JupyterLab

Create `notebooks/01_data_audit.ipynb` and run it top to bottom against `data/`
without training a model. The notebook should answer four questions.

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

## 2. Freeze the evaluation protocol

- Save a reproducible split manifest. Group by patient where IDs are available.
  Otherwise group by lesion and keep any linked duplicates together. Reserve an
  internal holdout before fitting anything, stratify by diagnosis where
  feasible, and inspect counts before choosing the number of development folds.
- Use validation folds for model selection, calibration, and thresholds. Agree
  with the biostatistics collaborator on what can be estimated with the
  available sample sizes, and report counts and uncertainty alongside scores.
- Compare models on the same lesions. Report a confusion matrix, macro F1 (each
  class gets equal weight), and per-class precision and recall. Inspect missed
  malignancies and performance by skin tone wherever counts support it. Record
  the decision rule, since ISIC's official macro F1 uses a per-class probability
  threshold of 0.5 and a local top-class prediction uses argmax.
  [Official evaluation protocol](https://challenge.isic-archive.com/landing/milk10k/#evaluation)

## 3. Train a clinical-photo baseline

Start with a majority-class baseline and a small pretrained image model with a
classifier head. Use the clinical close-up alone. Fine-tune if evaluation shows
room for improvement, then inspect errors for missed predictions and shortcuts.
The existing segmentation notebook is useful background. Whole-image
classification can get us to a baseline sooner, and segmentation is worth
revisiting if the error analysis shows the lesion is frequently missed.

Keep filenames, identifiers, and the diagnostic supplement out of model
features. Add available-at-capture metadata or dermoscopy only as controlled
comparisons on the same split. Report paired-image results under their own input
assumptions. Run the frozen model on the internal holdout and consider an ISIC
submission for comparison with the official benchmark.

## 4. Test the product assumption and build a private demo

- Look for independently labeled, self-taken phone photos from the intended
  population. Check label compatibility and measure the effect of device,
  lighting, skin tone, image quality, and unfamiliar conditions. Test how often
  the model abstains and how often it makes errors among the remaining cases.
  Choose any abstention rule on development data and verify it on independent
  data, including out-of-scope photos.
- Use curated, licensed examples in a local demo. Show the model's possible
  categories and an inconclusive state. A model trained on 11 known categories
  can return a high score for something outside all 11, and calibration on
  clinic images cannot establish reliability on self-taken photos.
- Treat public predictions on real user photos as a separate decision. Before
  pursuing that use, review the independent evidence with a dermatologist and
  address how people interpret the output, privacy for uploaded images, dataset
  permissions, and any applicable medical-device requirements.
  [IMDRF good machine learning practice principles](https://www.imdrf.org/documents/good-machine-learning-practice-medical-device-development-guiding-principles)
