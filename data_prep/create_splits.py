"""Freeze the MILK10k lesion-level test split and development folds."""

import argparse
import hashlib
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MANIFEST_NAME = "milk10k-v1.csv"
SEED = 42
SHUFFLES = 24
IMAGE_COLUMNS = {"clinical: close-up": "clinical_path", "dermoscopic": "dermoscopic_path"}
METADATA_COLUMNS = ["sex", "skin_tone_class", "age_approx", "site"]
BALANCE_COLUMNS = ["diagnosis_code", *METADATA_COLUMNS]


def load_lesions(data_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(data_dir / "MILK10k_Training_Metadata.csv", dtype={"age_approx": "Int64", "skin_tone_class": "Int64"})
    if metadata[["lesion_id", "isic_id", "image_type"]].isna().to_numpy().any() or not metadata["isic_id"].is_unique:
        raise ValueError("Image metadata needs complete keys and a unique isic_id per row")

    metadata["path"] = "MILK10k_Training_Input/" + metadata["lesion_id"] + "/" + metadata["isic_id"] + ".jpg"
    paths = metadata.pivot(index="lesion_id", columns="image_type", values="path")
    if set(paths.columns) != IMAGE_COLUMNS.keys() or paths.isna().to_numpy().any():
        raise ValueError("Expected one clinical close-up and one dermoscopic image per lesion")
    paths = paths.rename(columns=IMAGE_COLUMNS).rename_axis(columns=None)

    targets = pd.read_csv(data_dir / "MILK10k_Training_GroundTruth.csv").set_index("lesion_id")
    if not targets.index.is_unique:
        raise ValueError("Ground truth needs a unique lesion_id per row")
    if not targets.isin([0, 1]).to_numpy().all() or not targets.sum(axis=1).eq(1).all():
        raise ValueError("Expected one active diagnosis per lesion")
    if set(paths.index) != set(targets.index):
        raise ValueError("Metadata and ground-truth lesion IDs differ")

    clinical = metadata.loc[metadata["image_type"].eq("clinical: close-up")].set_index("lesion_id")
    labels = targets.idxmax(axis=1).rename("diagnosis_code")
    return paths.join(labels).join(clinical[METADATA_COLUMNS]).sort_index()


def duplicate_groups(lesions: pd.DataFrame, data_dir: Path) -> list[str]:
    """Connect lesion IDs that share an exact copy of either image."""
    parents = {lesion_id: lesion_id for lesion_id in lesions.index}

    def root(lesion_id: str) -> str:
        while parents[lesion_id] != lesion_id:
            lesion_id = parents[lesion_id]
        return lesion_id

    first_lesion_by_digest = {}
    for lesion_id, clinical, dermoscopic in lesions[["clinical_path", "dermoscopic_path"]].itertuples(name=None):
        for image_path in (clinical, dermoscopic):
            with (data_dir / image_path).open("rb") as image:
                digest = hashlib.file_digest(image, "sha256").digest()
            first_lesion = first_lesion_by_digest.setdefault(digest, lesion_id)
            parents[root(lesion_id)] = root(first_lesion)
    return [root(lesion_id) for lesion_id in lesions.index]


def choose_folds(lesions: pd.DataFrame, n_splits: int, *, holdout_only: bool = False) -> list[np.ndarray]:
    categories = lesions[BALANCE_COLUMNS].astype("string").fillna("Missing")
    indicators = pd.get_dummies(categories, dtype=int).to_numpy()
    target_rows = len(lesions) / n_splits
    target_counts = indicators.sum(axis=0) / n_splits

    by_diagnosis = pd.DataFrame({field: categories["diagnosis_code"] + "/" + categories[field] for field in ("sex", "skin_tone_class")})
    intersections = pd.get_dummies(by_diagnosis, dtype=int).to_numpy()
    # Aim to cover combinations that have at least one expected case per fold.
    intersections = intersections[:, intersections.sum(axis=0) >= n_splits]

    best_score = (float("inf"), float("inf"), float("inf"), float("inf"))
    best_folds = []
    for seed in range(SEED, SEED + SHUFFLES):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [validation for _, validation in splitter.split(lesions, lesions["diagnosis_code"], lesions["group_id"])]
        candidates = [[fold] for fold in folds] if holdout_only else [folds]
        for candidate in candidates:
            sizes = np.array([len(fold) for fold in candidate])
            counts = np.stack([indicators[fold].sum(axis=0) for fold in candidate])
            intersection_counts = np.stack([intersections[fold].sum(axis=0) for fold in candidate])
            score = (
                float(np.abs(sizes - target_rows).sum()),
                float(np.count_nonzero(counts == 0)),
                float(np.count_nonzero(intersection_counts == 0)),
                float(np.abs(counts - target_counts).sum()),
            )
            if score < best_score:
                best_score = score
                best_folds = candidate
    return best_folds


def assign_splits(lesions: pd.DataFrame) -> pd.DataFrame:
    (test_rows,) = choose_folds(lesions, n_splits=5, holdout_only=True)
    test_ids = lesions.index[test_rows]
    development = lesions.drop(index=test_ids)

    assignments = lesions[["diagnosis_code", "clinical_path", "dermoscopic_path", "group_id"]].copy()
    assignments["split"] = "development"
    assignments.loc[test_ids, "split"] = "test"
    assignments["validation_fold"] = pd.Series(pd.NA, index=assignments.index, dtype="Int64")
    for fold, rows in enumerate(choose_folds(development, n_splits=3), start=1):
        assignments.loc[development.index[rows], "validation_fold"] = fold
    return assignments


def save_manifest(manifest: pd.DataFrame, path: Path) -> None:
    content = manifest.to_csv(lineterminator="\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Publish the completed file atomically, with no replacement of a frozen version.
    with NamedTemporaryFile(dir=path.parent, delete_on_close=False) as temp:
        temp.write(content)
        temp.close()
        try:
            path.hardlink_to(temp.name)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Frozen manifest differs at {path}. Use a new manifest version for changed assignments.") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_dir)))
    data_dir = parser.parse_args().data_dir.expanduser().resolve()

    lesions = load_lesions(data_dir)
    lesions["group_id"] = duplicate_groups(lesions, data_dir)
    manifest = assign_splits(lesions)
    path = data_dir / "splits" / MANIFEST_NAME
    save_manifest(manifest, path)

    print(f"Manifest: {path}")
    print(manifest.groupby("split").size().to_string())
    print("\nDevelopment folds:")
    print(manifest["validation_fold"].value_counts().sort_index().to_string())
    for field in BALANCE_COLUMNS:
        categories = lesions[field].astype("string").fillna("Missing")
        print(f"\n{field}:")
        print(pd.crosstab(categories, manifest["split"]).to_string())


if __name__ == "__main__":
    main()
