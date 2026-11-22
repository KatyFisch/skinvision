# Classifier experiments and decisions

The retained workflow trains two classifiers on clinical close-ups from MILK10k. Both use
frozen DermFM-Zero-Open features and three logistic regression heads. Each head fits on
two development folds after removing the reserved calibration groups. The ensemble
uses the same three members to score calibration and holdout photos. The default
cutoff is set on calibration melanomas to target 95% sensitivity.

We kept **padded-crop MEL vs. NV** as the binary model because it met the melanoma
sensitivity target on the local holdout and is the model the demo already uses.
We kept **direct 11-class** as the all-diagnosis model because it met the same
target with the strongest macro F1 among the 11-class options that reached it.
Its decision rule flags MEL at the calibrated cutoff, then picks the highest-scoring
remaining diagnosis for photos below the cutoff. The app continues to accept
previously saved ResNet-18 cascade bundles for inference.

Full historical recipes and commands are in Git at `4ae2c1c`. The run IDs below
refer to local artifacts under the ignored `data/experiments/` directory.
[Train the retained models](../notebooks/02_train_classifiers.ipynb)

## MEL vs. NV and segmentation

The first masking comparison blacked out everything outside the predicted U-Net
mask. The original-photo model flagged 86 of 90 melanomas and correctly cleared 22
of 149 nevi. The black-background model flagged 78 of 90 and cleared 15 of 149.
It treated the 21 empty masks on holdout as inconclusive. In total, the U-Net
produced 86 empty masks across 1,196 lesions. Run `20261122T110813557377Z`.

The follow-up tested a crop around the lesion plus skin and a full-frame image
with a 12-pixel rim of skin at normal brightness and the remaining background
dimmed by 50%. Both fell back to the original frame on empty masks. The crop
pads by 25% of the longest bounding-box edge, with a 12-pixel minimum, then
fits a 4:3 crop and scales it to 600 × 450. Run
`20261122T112057117756Z-skin-context`.

| Input | MEL flagged at locked cutoff | NV cleared at locked cutoff | NV cleared at matched 86/90 MEL |
| --- | ---: | ---: | ---: |
| Original clinical photo | 86/90 | 22/149 | 28/149 |
| Dimmed background | 81/90 | 43/149 | 35/149 |
| Padded crop | 87/90 | 21/149 | 26/149 |

The padded crop performed close to the original frame. Dimming showed a
specificity gain at matched sensitivity, but its locked cutoff missed the target.
The matched comparison uses holdout labels to pick cutoffs retrospectively.
A clear advantage over the original image would require more evidence.

## Clinical photos across all 11 diagnoses

The 1,048-lesion local holdout included 90 melanomas and 958 other diagnoses.
Specificity here is the fraction of those 958 photos that did not receive a
melanoma flag. The 11-class macro F1 gives each diagnosis equal weight.
All nine variants used the same group-based calibration procedure and reached
the target on calibration.

| Frozen encoder | Head | MEL flagged | Specificity | MEL AUROC | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet-18 | Cascade | 86/90 | 26.8% | 0.681 | 0.093 |
| ResNet-18 | MEL vs. rest | 86/90 | 40.3% | 0.827 | |
| ResNet-18 | Direct 11-class | 88/90 | 47.5% | 0.853 | 0.163 |
| PanDerm Base | Cascade | 79/90 | 24.3% | 0.626 | 0.102 |
| PanDerm Base | MEL vs. rest | 87/90 | 56.3% | 0.857 | |
| PanDerm Base | Direct 11-class | 83/90 | 59.6% | 0.871 | 0.237 |
| DermFM-Zero-Open | Cascade | 82/90 | 71.4% | 0.876 | 0.341 |
| DermFM-Zero-Open | MEL vs. rest | 89/90 | 62.4% | 0.926 | |
| DermFM-Zero-Open | Direct 11-class | 86/90 | 67.7% | 0.924 | 0.323 |

The DermFM cascade had the highest macro F1 but missed the sensitivity target.
The DermFM MEL-vs-rest head flagged the most melanomas and gives a binary
decision. The direct head reached the target while emitting a diagnosis from
all 11 categories. The final DermFM run was `20261122T105620130348Z-dermfm_zero_open`.

## Earlier ResNet, ConvNeXt, and data experiments

- **Paired encoders and fine-tuning.** Frozen ResNet-18 and ConvNeXt-Tiny had
  24.3% and 21.4% nevus specificity at 95% sensitivity on the development folds.
  ConvNeXt had higher AUROC (0.789 vs. 0.772). Five epochs of ResNet fine-tuning
  raised AUROC to 0.817 with 24.0% specificity at that operating point.
  AUROC and specificity at the target can favor different recipes.
  Runs `20261121T135020843297Z` and `20261121T173906752089Z-resnet18-finetune`.
- **Clinical frame size.** ResNet-18 refits achieved 16.1% specificity for the
  224 × 224 center crop, 11.4% for the resized 341 × 256 full frame, and 21.1%
  for the native 600 × 450 full frame at 95% development sensitivity. The
  native full-frame refit was selected for the earlier clinical ResNet model.
  Runs `20261121T175752721371Z`, `20261121T183820515282Z`, and
  `20261121T183900179552Z`.
- **Augmentation.** Training with original and augmented clinical photos
  (brightness, contrast, and horizontal flips) gave no gain in this run.
  On the holdout, originals-only and augmented runs cleared 5 and 4 of 149
  nevi at their locked cutoffs and flagged all 90 melanomas. At a retrospectively
  matched 86 of 90 flagged melanomas, they cleared 37 and 36 nevi.
  Run `20261121T174112468028Z-resnet18-clinical-expanded-augmentation`.
- **Age.** Adding approximate age to paired-image features increased
  development specificity from 33.0% to 34.3% at 95% sensitivity. The gain was
  small and measured in one development comparison. The retained models
  use images alone. Run `age-20261121T174134385088Z`.
- **Extra ISIC 2017 dermoscopy.** A perceptual image screen excluded suspected
  cross-source duplicates and grouped suspected repeats before splitting.
  Adding ISIC 2017 raised pooled holdout AUROC from 0.733 to 0.776. At the locked
  cutoffs, sensitivity fell from 93.9% to 88.5%. At retrospectively matched
  sensitivity, specificity rose on ISIC 2017 and fell on MILK10k. The retained
  clinical heads train on MILK10k. Run `20261121T174146815189Z-isic2017-dermoscopy`.
- **Paired router and nine-class heads.** Refitting after shared ResNet-18
  fine-tuning improved nine-class development macro F1 from 0.300 to 0.341,
  with 0 of 7 `MAL_OTH` lesions identified. At 95% melanoma routing sensitivity,
  it sent 790 of 3,235 other lesions to the specialist, down from 1,221.
  Run `20261121T194734757774Z`.
- **Earlier clinical-only ResNet cascade.** The locked cascade flagged 89 of 90
  melanomas, but also flagged 871 of 958 non-melanomas. It achieved 9.1%
  specificity and 0.057 macro F1 on the 11-class holdout. This is the cascade
  format still supported by the app. Run `20261121T200623773355Z`.

## What these results cover

MILK10k ground truth is lesion-level. Most labels in the training release
were confirmed by histopathology, with some based on clinical assessment.
The frozen splits keep paired views and exact image copies together.
Patient IDs are unavailable, and exact matching can miss alternate views of
the same lesion. The U-Net was trained on ISIC 2017 dermoscopy.
DermFM-Zero-Open's published list of 38,404 ISIC IDs contained no exact match
for the MILK10k image IDs. Older archive IDs and other pretraining sources
can still contain related images. Some diagnoses have very few holdout
examples. These results use clinic photos and repeated comparisons on one
local holdout. Self-taken phone photos require independent evaluation.

[MILK10k dataset](https://api.isic-archive.com/doi/milk10k/) ·
[DermFM-Zero-Open checkpoint and pretraining sources](https://huggingface.co/redlessone/DermFM-Zero)
