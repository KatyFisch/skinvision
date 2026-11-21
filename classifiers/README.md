# Paired MEL-vs-NV training

From the repository root, run the comparison using the existing
[MEL-vs-NV manifest](../data_prep/README.md).

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
