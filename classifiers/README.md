# Lesion classifiers

Routine MILK10k training uses the full native **600 × 450** photos with the
checkpoint's RGB normalization. The selected clinical MEL-vs-NV variant is
the native-resolution, five-epoch run with a refitted head. Its three local
development checkpoints are in `data/models/clinical-mel-nv/`.

The selected native paired baselines are in `data/models/mel-nv-router/` and
`data/models/other-diagnoses/`. Each directory has the three fitted fold heads
and metrics for the shared run. Each of these heads takes features from both
the clinical and dermoscopic photo.

## Use the preprocessed image catalog

Pass `--catalog final_data.csv` to the classifier commands to use the export from
`preprocessing/preprocess.py`. Use `--data-dir /path/to/data` for a shared
dataset. Both the catalog filename and the image paths in it resolve relative
to that directory. You can also pass an absolute catalog filename.

```sh
uv run --locked python -m classifiers.train_mel_nv --data-dir /path/to/data --catalog final_data.csv
```

The catalog needs one row per image with `dataset`, `image_id`, `lesion_id`,
`image_type`, `path`, `age`, and either `diagnosis_code` or the `MEL` and `NV`
flags for MEL-vs-NV. The routed tasks need the original MILK10k diagnosis as
`diagnosis_code` or as one of the 11 one-hot flags. Use the dataset values
`MILK10k` and `ISIC2017`. MILK10k needs one `clinical: close-up` and one
`dermoscopic` image per lesion. The loader treats ISIC 2017 rows with blank
`image_type` as dermoscopic. Those rows can leave `lesion_id` empty. Age is
optional per image and only used by the age comparison, which reads it from
the clinical view.

The frozen manifests still assign lesions and duplicate groups to folds and
holdouts. The loader resolves catalog images for those assignments and checks
that their identities and diagnosis labels agree. ISIC rows outside the eligible
manifest, including seborrheic keratoses and suspected cross-dataset matches,
stay outside the experiment.

Runs record the catalog checksum. When comparing age or scoring checkpoints
from a catalog-backed run, pass `--catalog` with the same CSV. Omit the flag
to use the original manifest paths for runs made without a catalog.

## Train the router and other-diagnosis classifier

Generate the two [routed task manifests](eval_split/README.md#routed-milk10k-tasks)
from the catalog and the frozen parent MILK10k split. Both use paired images
and inherit the specialist's fold and holdout assignments wherever they share
lesions. Keep the original `diagnosis_code` alongside the model target for
evaluating per-diagnosis errors.

```sh
uv run --locked python -m classifiers.train_routed --catalog final_data.csv
```

The baseline fits two logistic-regression heads on shared frozen ResNet-18
features from a native-resolution clinical and dermoscopic image per lesion.
It trains each head inside the development folds and writes the fitted heads,
predictions, and metrics under `data/experiments/routed/`. Use `--data-dir` for
shared data.

- **Router.** Train on every MILK10k lesion. The positive class is `MEL_OR_NV`
  (1), and the negative class is `OTHER` (0). Choose the exploratory cutoff
  based on melanoma routing sensitivity on the development predictions
  (95% by default). The report shows how many lesions of each diagnosis
  reach the specialist.
- **Other-diagnosis classifier.** Train on the true `OTHER` subset and predict
  its nine diagnosis codes. Compare to a BCC majority baseline. Report macro
  F1, per-class recall with counts, and a confusion matrix. Evaluate the
  complete routing pipeline separately using the same held-out lesions.

### Fine-tune the shared encoder

```sh
uv run --locked python -m classifiers.finetune_routed --catalog final_data.csv
```

This starts each fold from the frozen logistic heads and trains both heads
and ResNet-18's last block for five epochs. The router loss uses all training
lesions. The nine-class loss uses true `OTHER` lesions. The encoder is shared
by the two tasks. The command reuses a local float32 cache of the frozen
layers for the native photos, which takes about 9.4 GB for MILK10k.

The report compares the frozen baseline, the joint fine-tune, and fresh
logistic heads fit on the fine-tuned features. It saves fold checkpoints,
development predictions, and metrics under `data/experiments/routed-finetune/`.

In the three MILK10k development folds (`20261121T194734757774Z`), the refit
gave the best results. At 95% melanoma routing sensitivity (342 of 360), the
router sent 790 of 3,235 Other lesions to the specialist, compared with 1,221
for the frozen encoder. Nine-class macro F1 rose from 0.300 to 0.341, and
accuracy rose from 62.8% to 70.0%. The refit correctly classified 0 of 7
`MAL_OTH` lesions. These figures use development predictions, with the
separate test set reserved for later evaluation.

## Run the single-photo pipeline

Train the router and nine-class head with one native clinical close-up per
lesion. The evaluation reuses the existing clinical MEL-vs-NV specialist.

```sh
uv run --locked python -m classifiers.finetune_routed --clinical-only --catalog final_data.csv
uv run --locked python -m classifiers.single_photo evaluate --routed-run experiments/routed-finetune-clinical/RUN_ID
```

Use the routed run ID printed by the training command. Pass `--specialist-run`
to use a clinical MEL-vs-NV training run directly. Evaluation aims to route
at least 99% of development melanomas to the specialist and flag at
least 95% of them end to end. It locks both cutoffs from out-of-fold
predictions, then averages the three fold models' scores on the existing
20% labeled MILK10k holdout. It writes `model.json`, predictions, and metrics
under `data/experiments/single-photo/`. The report counts where melanomas
were missed and includes an 11-class confusion matrix. The 95% figure is a
development target. Holdout scores use an ensemble and may reach a different
operating point at the same cutoffs.

Run `20261121T200623773355Z` flagged 89 of 90 melanomas on the 1,048-lesion
holdout. The router sent all 90 onward and the specialist missed one. The
pipeline also flagged 871 of 958 non-melanoma lesions, giving 9.1% melanoma
specificity. Eleven-class accuracy was 14.9% and macro F1 was 0.057. The
cutoffs were locked before scoring these clinic photos.

Run the saved model locally with a curated example. The command reads one
image, applies its EXIF orientation, and preserves the full frame. If the
longest edge exceeds 600 pixels it scales the photo down proportionally.
The model emits one research label from the selected branch.

```sh
uv run --locked python -m classifiers.single_photo predict --model data/experiments/single-photo/RUN_ID/model.json path/to/example.jpg
```

For a useful close-up, use even daylight, keep the whole lesion and a margin
of surrounding skin in view, and tap the lesion to focus. Back up slightly
if the camera cannot focus. Review the image for blur, glare, or shadow and
use the original JPEG or PNG without filters or digital zoom. For a clinician,
an overview of the body site and a close-up with a ruler beside the lesion
can also help. [American Academy of Dermatology photo guide](https://www.aad.org/public/fad/digital-health/taking-pictures-skin)

MILK10k close-ups are clinic photos. This run measures how the pipeline
works on those photos. Independent evaluation on self-taken photos and
clinical review are needed before using it for people's skin concerns.

## Melanoma score bands

The local demo describes the masked cascade's raw MEL-vs-NV specialist score
as below cutoff, low, moderate, or high. Below cutoff uses the saved model's
specialist cutoff. Illustrative score cutoffs of `0.50` and `0.90` start the
moderate and high bands. The holdout used the same three-fold ensemble as the
demo and reused the U-Net masks from the [masking comparison](#compare-masking-for-mel-vs-nv).

| Band | Masked holdout melanomas | Masked holdout nevi |
| --- | ---: | ---: |
| Below cutoff | 1 | 0 |
| Low | 6 | 11 |
| Moderate | 24 | 53 |
| High | 52 | 71 |

The table covers the 218 holdout photos with nonempty masks that reached the
specialist. It excludes seven melanomas and 14 nevi with empty masks and other
diagnosis classes. These bands summarize the model score for the private demo.
Clinical probabilities require calibration and validation on labeled photos
representative of the intended use.

## Compare masking for MEL-vs-NV

```sh
uv run --locked python -m classifiers.compare_masked_mel_nv --segmentation-model models/unet_segmentation_full.pth
```

Use the same U-Net checkpoint as the demo. For each native 600 × 450 MILK10k
clinical photo of a melanoma or nevus, compare its original pixels with the
demo's black-background composite from the predicted mask. Both variants use
frozen DermFM-Zero-Open features and the same logistic regression recipe. Both
train on the same development lesions with nonempty masks. Each uses the
three-member ensemble to set a cutoff on the same 189 calibration lesions,
then scores the original 239-lesion holdout. Empty masks count as inconclusive
in the masked variant and stay in the end-to-end denominators.

| Input | Holdout melanomas flagged | Holdout nevi correctly labeled NV | Inconclusive |
| --- | ---: | ---: | ---: |
| Original clinical photo | 86/90 (95.6%) | 22/149 (14.8%) | 0 |
| Black background outside the predicted mask | 78/90 (86.7%) | 15/149 (10.1%) | 21 |

The U-Net produced empty masks for 86 of 1,196 lesions across all splits,
including seven melanomas and 14 nevi on the holdout. That limits the masked
pipeline's holdout sensitivity to 83 of 90 even with perfect scoring on
nonempty masks. On the shared subset with nonempty masks, compare the heads
at the same 79 of 83 flagged melanomas. The original photos cleared 19 of
135 nevi and the masked photos cleared 15. These cutoffs use holdout labels
for a retrospective comparison.

Use original pixels for the classifier candidate in the private demo. The
segmentation overlay can still show the predicted lesion. The U-Net was
trained on ISIC 2017 dermoscopy according to the segmentation notebook.
An earlier cross-dataset image screen found no suspected ISIC 2017 match in
these held-out lesions among the cases it screened. Mask quality and classifier
performance on self-taken photos need their own evaluation.

Runs save the metrics, heads, mask coverage, masked photos, and predictions
under `data/experiments/masked-mel-nv/`. The original DermFM-Zero-Open feature
cache can be reused.

## Compare three single-photo classifier designs

```sh
uv run --locked python -m classifiers.compare_clinical --encoder resnet18
uv run --locked python -m classifiers.compare_clinical --encoder panderm_base
```

Use `--data-dir` to point to an existing MILK10k catalog and frozen splits.
The commands cache embeddings from the same native 600 × 450 clinical
photos. PanDerm Base uses the published ViT-B/16 weights, with positional
interpolation and zero padding to complete edge patches. The encoder stays
frozen. [PanDerm checkpoint and license](https://github.com/SiyuanYan1/PanDerm#1-download-panderm-pre-trained-weights) ·
[PanDerm paper and pretraining sources](https://doi.org/10.1038/s41591-025-03747-y)

To include DermFM-Zero-Open, accept the research terms on the model page and
sign in locally with a token that can read the gated repository:

```sh
uv run --locked hf auth login
uv run --locked python -m classifiers.compare_clinical --encoder dermfm_zero_open
```

The run uses the frozen vision encoder's 1,024 pooled features before the
projection used for text matching. It retains the full clinical frame and
uses the release's RGB normalization, bicubic positional interpolation, and
zero padding at the image edges. The checkpoint revision and published ISIC
image list are verified by checksum. Weights and the image list are cached
locally under `data/checkpoints/dermfm-zero/`.
[DermFM-Zero-Open weights and terms](https://huggingface.co/redlessone/DermFM-Zero) ·
[Authors' linear-probe implementation](https://github.com/SiyuanYan1/DermFM-Zero/blob/c9f3d6a149ab048ae8a98ea4701e23b271e7368a/src/open_clip/utils.py)

The experiment compares a refitted cascade, a binary head trained on melanoma
and all other diagnoses, and a direct 11-class head. Every head uses regularized
logistic regression. A deterministic group-stratified sample of 825 development
lesions (72 melanomas) calibrates the cutoffs. The three ensemble members each
train on the non-calibration groups from two frozen folds. The same fitted
members and probability averaging score calibration and the 1,048-lesion
holdout. All nine configurations flagged 69 of 72 calibration melanomas at the 95%
target. The cascade retains its 99% routing target. The 11-class decisions
use the melanoma cutoff, then the highest-scoring other class for unflagged
lesions.

| Frozen encoder | Classifier | Holdout melanomas flagged | Specificity | Melanoma AUROC | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| ResNet-18 | Cascade | 86/90 | 26.8% | 0.681 | 0.093 |
| ResNet-18 | Melanoma vs. rest | 86/90 | 40.3% | 0.827 | — |
| ResNet-18 | Direct 11-class | 88/90 | 47.5% | 0.853 | 0.163 |
| PanDerm Base | Cascade | 79/90 | 24.3% | 0.626 | 0.102 |
| PanDerm Base | Melanoma vs. rest | 87/90 | 56.3% | 0.857 | — |
| PanDerm Base | Direct 11-class | 83/90 | 59.6% | 0.871 | 0.237 |
| DermFM-Zero-Open | Cascade | 82/90 | 71.4% | 0.876 | 0.341 |
| DermFM-Zero-Open | Melanoma vs. rest | 89/90 | 62.4% | 0.926 | — |
| DermFM-Zero-Open | Direct 11-class | 86/90 | 67.7% | 0.924 | 0.323 |

At the locked cutoffs, DermFM-Zero-Open's melanoma-vs-rest head flagged two
additional melanomas and 59 fewer non-melanomas than the PanDerm Base version.
The direct 11-class head also reached the 95% sensitivity target and raised
macro F1 from 0.237 to 0.323. The DermFM-Zero-Open cascade had the highest macro
F1 of these runs and reached 91.1% melanoma sensitivity. At a retrospectively
matched sensitivity of 86 of 90 melanomas, specificity for the
DermFM-Zero-Open cascade, binary head, and direct 11-class head was 70.4%,
71.5%, and 69.4%, respectively. This ranking comparison uses holdout labels to
choose the operating points.

PanDerm's pretraining includes ISIC Archive data. MILK10k includes lesions
previously archived in ISIC, and available provenance does not establish
whether earlier images of these lesions were used in the checkpoint. Treat the
comparison as exploratory. The local holdout is also from the same clinic-photo
release used for development. None of the 38,404 ISIC IDs in DermFM-Zero-Open's
published list exactly matched the current MILK10k clinical or dermoscopic
image IDs. Older archive identifiers and the other pretraining sources can
still contain related images.
[MILK10k dataset provenance](https://api.isic-archive.com/doi/milk10k/) ·
[DermFM-Zero-Open pretraining sources](https://huggingface.co/redlessone/DermFM-Zero#model-details)

Each run writes local `heads.joblib`, calibration and holdout predictions,
and `metrics.json` with cutoffs, provenance, and per-class recall under
`data/experiments/clinical-comparison/`. The demo uses its existing fitted
cascade.

## Compare frozen paired-image encoders

From the repository root, run the comparison using the existing
[MEL-vs-NV manifest](eval_split/README.md).

```sh
uv run --locked python -m classifiers.train_mel_nv
```

Pass `--data-dir /path/to/data` when using a shared dataset from a worktree. Use
`--encoder resnet18` or `--encoder convnext_tiny` to run one encoder.
`--sensitivity-target` controls the exploratory target and defaults to 0.95.
Use `--center-crop` to reproduce the historical 224 × 224 comparison.

Both pretrained encoders stay frozen. The script concatenates the clinical and
dermoscopic features and fits the same regularized logistic regression recipe on
the same three folds. It fits feature scaling inside each training fold. The
score for each development lesion comes from the fold that held it out. Each
threshold is selected on these development scores, which also describe the
resulting sensitivity and specificity. The test holdout stays reserved for a
later evaluation of the selected model and threshold.

Checkpoints are saved model weights. Here the checkpoints are the official
ImageNet-1K V1 weights for ResNet-18 and ConvNeXt-Tiny. The script verifies and
records the full SHA-256 hashes and uses each checkpoint's RGB normalization.
Verify the publisher, license, training data, and preprocessing
before trying a new checkpoint. Keep track of any possible overlap with
evaluation images.

Weights, cached features, and run outputs live under the ignored dataset
directory. Each run saves `metrics.json` and `validation_predictions.csv` in
`experiments/mel-nv/<run-id>/`.

## Fine-tune the paired ResNet-18

```sh
uv run --locked python -m classifiers.finetune_mel_nv
```

Each fold runs two passes using only its training lesions:

1. Fit the same logistic regression head on the frozen paired features.
2. Load that head into PyTorch and update it along with ResNet's last block
   (`layer4`) for five epochs. The head uses a learning rate of `0.001` and the
   block uses `0.0001`. Earlier layers and BatchNorm running statistics stay
   fixed. Both passes use the same native-resolution inputs.

The run compares the frozen baseline and checkpoints after epochs 1, 2, and 5
on the same held-out development folds. With `--epochs`, the final checkpoint
is saved at the requested epoch. The overall metrics use one cutoff from the
pooled development predictions for each checkpoint and sensitivity target.
Per-fold metrics use a separate exploratory cutoff for each fold. The run
selects the variant with the highest pooled specificity at the target
sensitivity, preferring the earliest variant in a tie. Scores, training
losses, and fold checkpoints are saved under
`data/experiments/mel-nv/<run-id>-resnet18-finetune-full-frame-native/`.

## Fine-tune on clinical photos only

```sh
uv run --locked python -m classifiers.finetune_mel_nv --catalog final_data.csv --clinical-only
```

This uses one clinical close-up per lesion and the same development folds,
learning rates, and five-epoch recipe. After the last epoch, fit a fresh
logistic-regression head on the fine-tuned features from that fold's training
photos. The report compares frozen, fine-tuned, and refitted scores on the
held-out development folds. Look for the `-resnet18-clinical-finetune-full-frame-native`
run in the experiments directory.

To compare the standard center crop with the same resize and the full frame,
run both commands on the same catalog and development folds.

```sh
uv run --locked python -m classifiers.finetune_mel_nv --catalog final_data.csv --clinical-only --center-crop
uv run --locked python -m classifiers.finetune_mel_nv --catalog final_data.csv --clinical-only --full-frame
```

The second run keeps the resized 341 × 256 MILK10k frames. The resize and
normalization match the standard ResNet transform. Batching uses the common
4:3 aspect ratio of these clinical photos. Compare the same head variant in
each run at the target melanoma sensitivity. The full-frame run has a separate
feature cache entry and a `-full-frame` directory suffix.

To isolate resolution, compare that resized full-frame run with a native
600 × 450 run using the same catalog and training options.

```sh
uv run --locked python -m classifiers.finetune_mel_nv --catalog final_data.csv --clinical-only --full-frame --native-resolution
```

The native-resolution run keeps the decoded photo dimensions and applies the
same RGB normalization. It writes a separate `-full-frame-native` run.

The [historical clinical augmentation experiment](augmentation/README.md)
compares brightness, contrast, and horizontal flips against the saved
center-crop baseline on the same development folds and MILK10k holdout.

## Compare images with and without age

```sh
uv run --locked python -m classifiers.compare_age data/experiments/mel-nv/your-run-id-resnet18-finetune
```

Pass the path printed by the fine-tuning run. For each development fold, use the
selected variant's encoder to extract the views used during training. Use `--variant fine_tuned`
to compare the final checkpoint when another variant was selected. Fit two logistic
regression heads on those same training features, one with age appended. Use
the training-fold median for missing ages when fitting the age head. At
prediction time, use the image-only score when age is missing. The report
includes per-fold results and missed melanoma counts by age group. It lives in
an `age-<run-id>/` subdirectory of the source run.

## Test whether ISIC 2017 training data help

Generate the [ISIC 2017 split](eval_split/README.md#isic-2017-mel-vs-nv-split)
first. Train both recipes with one dermoscopic photo per example.

```sh
uv run --locked python -m classifiers.compare_isic2017
```

For each MILK10k development fold, train on the other two folds, then repeat
with the ISIC 2017 training split added. Both recipes use the same ResNet-18
setup, five epochs of last-block fine-tuning, and a logistic head refit on the
resulting embeddings. An epoch covers the whole training set, so the combined
recipe takes more optimizer steps. Both recipes use image features alone.

The run reports out-of-fold MILK10k development scores. For ISIC 2017, average
the three fold models and choose a development cutoff per recipe at 95%
sensitivity on the ISIC validation split. Use the saved checkpoints and cutoffs
to score the ISIC test split in a separate command. Pass the run directory
printed by training.

```sh
uv run --locked python -m classifiers.compare_isic2017 --test-run data/experiments/mel-nv/your-run-id-isic2017-dermoscopy
```

The run keeps validation and test reports under `data/experiments/mel-nv/`.
Add `--include-milk-test` to also score the local MILK10k holdout without
changing the checkpoints or ISIC validation cutoffs. The combined reports
keep metrics by dataset and pool the two test sets. They also report the
exploratory comparison at matched sensitivity, which uses test labels to find
the comparison cutoffs.
