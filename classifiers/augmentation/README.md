# Clinical photo augmentation

Run the comparison from the repository root with the saved five-epoch
clinical-only ResNet-18 run. The command uses the same MILK10k manifest and
original image paths as that baseline.

```sh
uv run --locked python -m classifiers.augmentation.compare_clinical data/experiments/mel-nv/20261121T164756831394Z-resnet18-clinical-finetune
```

Use `--data-dir` for a shared dataset, or `--device` to override the device
recorded by the baseline.

Each of the 957 development lesions gets one lossless PNG per epoch. The
script independently samples brightness and contrast factors from 0.9 to 1.1,
applies them in that order, and flips the photo horizontally with probability
0.5. The seed and per-photo parameters make the image variants reproducible.
The assets and their source manifest, `assets.csv`, live in
`data/augmented/milk10k-clinical-mel-nv/<run-id>/`.

Each epoch shuffles 638 original training photos together with 638 augmented
copies in each fold. The model sees both views of every eligible training
lesion during each epoch. The initial logistic head fits the original training
photos, as in the baseline. Both runs fine-tune the last block for five epochs.
The expanded training set takes 80 optimizer steps per epoch. The baseline
takes 40 with the same batch size. Development and holdout scores use original
clinical photos with the same ImageNet preprocessing.
The labels and lesion groups come from the [frozen MILK10k split](../eval_split/README.md).

The development report pools the predictions from the fold that held each
lesion out to set each model's cutoff at 95% melanoma sensitivity. The holdout
report averages the three fold probabilities per lesion and applies those
saved cutoffs. Averaging can change score calibration, so the sensitivity
achieved on the holdout may differ from 95%. The report also gives specificity
at matched sensitivity using cutoffs computed from the holdout labels. This
describes the trade-off on these photos and does not establish a future cutoff.

The checkpoints, metrics, and per-lesion predictions live in
`data/experiments/mel-nv/<run-id>-resnet18-clinical-expanded-augmentation/`.
Both output directories are ignored by Git. The experiment evaluates MILK10k
clinical close-ups and uses a single training seed.
