# MEL-vs-NV training

## Use the preprocessed image catalog

Pass `--catalog final_data.csv` to any command below to use the export from
`preprocessing/preprocess.py`. Use `--data-dir /path/to/data` for a shared
dataset. Both the catalog filename and the image paths in it resolve relative
to that directory. You can also pass an absolute catalog filename.

```sh
uv run --locked python -m classifiers.train_mel_nv --data-dir /path/to/data --catalog final_data.csv
```

The catalog needs one row per image with `dataset`, `image_id`, `lesion_id`,
`image_type`, `path`, `age`, and either `diagnosis_code` or the `MEL` and `NV`
flags. Use the dataset values `MILK10k` and `ISIC2017`. MILK10k needs one
`clinical: close-up` and one `dermoscopic` image per lesion. ISIC 2017 rows use
`dermoscopic` and can leave `lesion_id` empty. Age is optional per image and
only used by the age comparison, which reads it from the clinical view.

The frozen manifests still assign lesions and duplicate groups to folds and
holdouts. The loader resolves catalog images for those assignments and checks
that their identities and MEL/NV labels agree. ISIC rows outside the eligible
manifest, including seborrheic keratoses and suspected cross-dataset matches,
stay outside the experiment.

Runs record the catalog checksum. When comparing age or scoring checkpoints
from a catalog-backed run, pass `--catalog` with the same CSV. Omit the flag
to use the original manifest paths for runs made without a catalog.

## Compare frozen paired-image encoders

From the repository root, run the comparison using the existing
[MEL-vs-NV manifest](eval_split/README.md).

```sh
uv run --locked python -m classifiers.train_mel_nv
```

Pass `--data-dir /path/to/data` when using a shared dataset from a worktree. Use
`--encoder resnet18` or `--encoder convnext_tiny` to run one encoder.
`--sensitivity-target` controls the exploratory target and defaults to 0.95.

Both pretrained encoders stay frozen. The script concatenates the clinical and
dermoscopic features and fits the same regularized logistic regression recipe on
the same three folds. It fits feature scaling inside each training fold. The
score for each development lesion comes from the fold that held it out. Each
threshold is selected on these development scores, which also describe the
resulting sensitivity and specificity. The test holdout stays reserved for a
later evaluation of the selected model and threshold.

Checkpoints are saved model weights. Here the checkpoints are the official
ImageNet-1K V1 weights for ResNet-18 and ConvNeXt-Tiny. The script verifies and
records the full SHA-256 hashes and uses each checkpoint's documented
preprocessing. Verify the publisher, license, training data, and preprocessing
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
   fixed. Both passes use the checkpoint's original image preprocessing.

The run compares the frozen baseline and checkpoints after epochs 1, 2, and 5
on the same held-out development folds. With `--epochs`, the final checkpoint
is saved at the requested epoch. The overall metrics use one cutoff from the
pooled development predictions for each checkpoint and sensitivity target.
Per-fold metrics use a separate exploratory cutoff for each fold. The run
selects the variant with the highest pooled specificity at the target
sensitivity, preferring the earliest variant in a tie. Scores, training
losses, and fold checkpoints are saved under
`data/experiments/mel-nv/<run-id>-resnet18-finetune/`.

## Fine-tune on clinical photos only

```sh
uv run --locked python -m classifiers.finetune_mel_nv --clinical-only
```

This uses one clinical close-up per lesion and the same development folds,
learning rates, and five-epoch recipe. After the last epoch, fit a fresh
logistic-regression head on the fine-tuned features from that fold's training
photos. The report compares frozen, fine-tuned, and refitted scores on the
held-out development folds. Look for the `-resnet18-clinical-finetune` run in
the experiments directory.

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
