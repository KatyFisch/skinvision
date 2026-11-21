# SkinVision project plan

The goal is an educational app that suggests possible skin lesion diagnoses from
an ordinary phone photo. Develop three classifiers on the paired MILK10k images
first. The app will use one to route a photo to either a MEL-vs-NV specialist or
a classifier covering the nine other MILK10k diagnoses.

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

## 2. Use one split across all three classifiers

- Use the [MEL-vs-NV manifest](classifiers/eval_split/README.md) for the specialist.
  Prepare the router and other-diagnosis manifests from the preprocessed image
  catalog and the frozen parent MILK10k assignments. The 20% holdout and three
  development folds stay fixed across all three tasks.
- Keep both views of each lesion and known duplicates together. Group by patient
  if IDs become available. Fit models and choose thresholds using the
  development folds. Evaluate the finalized models and routing rules on the
  held-out lesions.

## 3. Train the three paired-image classifiers

1. **MEL vs. NV specialist.** Train on melanomas (`MEL`) and melanocytic nevi
   (`NV`). Agree on an exploratory melanoma sensitivity target with the
   biostatistics collaborator and compare specificity against nevi at that
   target. Report missed melanomas, flagged nevi, and uncertainty.
2. **MEL-or-NV vs. other router.** Use every MILK10k diagnosis. Combine `MEL`
   and `NV` into one target and map the other nine diagnoses to `Other`. Measure
   how often melanomas and nevi reach the specialist, and which other diagnoses
   are mistakenly sent there.
3. **Other-diagnosis classifier.** Train it on the nine MILK10k categories
   outside `MEL` and `NV`, and use it for cases routed to `Other`. Report macro
   F1, per-class recall with case counts, and a confusion matrix.

Start each task with a majority-class reference and a small pretrained image
encoder. Combine features from the clinical and dermoscopic views into one
prediction per lesion. Use the full native-resolution frame when training the
specialist, router, and other-diagnosis classifier. Keep filenames,
identifiers, and the diagnostic supplement out of model features. Fine-tune
or revisit segmentation when validation errors give a reason to try it.

Evaluate approximate age as an optional input for the MEL-vs-NV specialist.
Compare images alone with images plus age on the same development folds, and
report melanoma sensitivity by age group. Use the image-only model as the
fallback when age is omitted. Use recorded sex for dataset audits only.

When the separate MEL/NV dataset is available, check usage terms, label
definitions, image modalities, and overlap with the MILK10k holdout. Compare the
specialist trained on MILK10k alone with a version that uses the additional
data. Keep source-specific evaluation so we can see whether any gain transfers
between datasets.

## 4. Route and evaluate the full pipeline

- Run the router first. Send `MEL-or-NV` cases to the specialist and `Other`
  cases to the other-diagnosis classifier. The `Other` group includes benign
  conditions and cancers such as basal cell carcinoma.
- Tune routing on the development folds and test an inconclusive response for
  uncertain assignments.
- Measure melanoma sensitivity from input to final output on the holdout.
  Track whether missed melanomas came from routing or classification, and
  report how often the system returns an inconclusive result. Inspect results
  by skin tone wherever counts support it.

## 5. Test the phone-photo workflow and build a private demo

- The [single-photo pipeline](classifiers/README.md#run-the-single-photo-pipeline)
  uses clinical-only MILK10k models for all three stages. The first holdout
  run flagged 89 of 90 melanomas and 871 of 958 non-melanomas at the locked
  development cutoffs. Improve specificity and evaluate the pipeline
  separately on self-taken photos.
- Carry forward the clinical MEL-vs-NV model trained on the original
  600 × 450 photos, with five epochs of fine-tuning and a refitted head.
  The three selected development checkpoints are in
  `data/models/clinical-mel-nv/`.
  In the earlier center-crop augmentation experiment, the originals-only run
  flagged 112 of 149 nevi and the expanded augmentation run flagged 113 at
  the same sensitivity of 86 out of 90 melanomas.
  [Clinical augmentation experiment](classifiers/augmentation/README.md)
- Look for independently labeled, self-taken phone photos from the intended
  population. Check label compatibility and measure the effect of device,
  lighting, skin tone, image quality, and unfamiliar conditions. Test how often
  the model abstains and how often it makes errors among the remaining cases.
  Choose any abstention rule on development data and verify it on independent
  data, including out-of-scope photos.
- Use curated, licensed examples in a local demo. Show the model's possible
  categories and an inconclusive state. Include melanoma look-alikes and other
  lesion types in later evaluations of the phone-photo workflow.
- Treat public predictions on real user photos as a separate decision. Before
  pursuing that use, review the independent evidence with a dermatologist and
  address how people interpret the output, privacy for uploaded images, dataset
  permissions, and any applicable medical-device requirements.
  [IMDRF good machine learning practice principles](https://www.imdrf.org/documents/good-machine-learning-practice-medical-device-development-guiding-principles)
