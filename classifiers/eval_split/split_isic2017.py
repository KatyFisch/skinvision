"""Freeze local ISIC 2017 MEL/NV splits after screening for image overlap."""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from classifiers.eval_split.create_splits import SEED, save_manifest

MANIFEST = "isic2017-mel-nv-v1.csv"
OVERLAP = "isic2017-mel-nv-overlap-v1.csv"
LABELS = "ISIC-2017_Training_Part3_GroundTruth.csv"
IMAGE_DIR = "ISIC-2017_Training_resizedImages"
MILK_METADATA = "MILK10k_Training_Metadata.csv"


def load_isic2017(data_dir: Path) -> pd.DataFrame:
    images = pd.read_csv(data_dir / LABELS).set_index("image_id").sort_index()
    flags = images[["melanoma", "seborrheic_keratosis"]]
    if images.index.hasnans or not images.index.is_unique or not flags.isin([0, 1]).to_numpy().all() or flags.sum(axis=1).gt(1).any():
        raise ValueError("Expected unique ISIC image IDs and valid Task 3 diagnosis flags")
    images["diagnosis_code"] = np.select([images.melanoma.eq(1), images.seborrheic_keratosis.eq(1)], ["MEL", "SK"], default="NV")
    images["image_path"] = IMAGE_DIR + "/" + images.index + ".jpg"
    return images[["diagnosis_code", "image_path"]]


def thumbnails(data_dir: Path, paths: pd.Series) -> np.ndarray:
    """Compare the center at two scales to accommodate crop and export differences."""
    samples = []
    for path in paths:
        with Image.open(data_dir / path) as photo:
            square = ImageOps.fit(ImageOps.exif_transpose(photo).convert("L"), (80, 80), Image.Resampling.LANCZOS)
            samples.append([np.asarray(view.resize((48, 48), Image.Resampling.LANCZOS)) for view in (square, square.crop((6, 6, 74, 74)))])
    return np.asarray(samples)


def image_hashes(images: np.ndarray) -> np.ndarray:
    squares = images.reshape(-1, 48, 48).astype(np.float32)
    low_frequencies = np.stack([cv2.dct(square)[:8, :8].ravel() for square in squares])
    bits = low_frequencies > np.median(low_frequencies[:, 1:], axis=1, keepdims=True)
    return np.packbits(bits, axis=1).view(np.uint64).reshape(images.shape[:-2])


def matching_images(left: np.ndarray, right: np.ndarray, *, within_collection: bool = False) -> list[tuple[int, int, float]]:
    """Flag near-identical views using both perceptual hashes and pixel agreement."""
    right_hashes = image_hashes(right)
    matches = {}
    for flipped in (False, True):
        oriented = left[..., ::-1] if flipped else left
        for turns in range(4):
            rotated = np.rot90(oriented, turns, axes=(-2, -1))
            left_hashes = image_hashes(rotated)
            for index, hashes in enumerate(left_hashes):
                start = index + 1 if within_collection else 0
                other, right_scale, left_scale = np.where(np.bitwise_count(right_hashes[start:, :, None] ^ hashes) <= 8)
                for candidate, scale_a, scale_b in zip(other + start, left_scale, right_scale, strict=True):
                    a, b = rotated[index, scale_a], right[candidate, scale_b]
                    # Strong agreement across the full thumbnail and its central region avoids matching similar dark borders.
                    full = cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).item()
                    center = cv2.matchTemplate(a[8:-8, 8:-8], b[8:-8, 8:-8], cv2.TM_CCOEFF_NORMED).item()
                    if full >= 0.97 and center >= 0.97:
                        key = (index, int(candidate))
                        matches[key] = max(matches.get(key, 0.0), min(full, center))
    return [(a, b, score) for (a, b), score in sorted(matches.items())]


def split_isic2017(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = load_isic2017(data_dir)
    milk = pd.read_csv(data_dir / MILK_METADATA, usecols=["lesion_id", "isic_id", "image_type"])
    dermoscopy = milk.loc[milk.image_type.eq("dermoscopic")].reset_index(drop=True)
    milk_paths = "MILK10k_Training_Input/" + dermoscopy.lesion_id + "/" + dermoscopy.isic_id + ".jpg"
    print("Reading ISIC 2017 and MILK10k dermoscopy thumbnails", flush=True)
    isic_thumbnails = thumbnails(data_dir, images.image_path)
    milk_thumbnails = thumbnails(data_dir, milk_paths)
    print("Screening for repeated images", flush=True)
    cross = matching_images(isic_thumbnails, milk_thumbnails)
    within = matching_images(isic_thumbnails, isic_thumbnails, within_collection=True)

    first = [a for a, _, _ in within]
    second = [b for _, b, _ in within]
    graph = csr_matrix((np.ones(len(within)), (first, second)), shape=(len(images), len(images)))
    _, group_ids = connected_components(graph, directed=False)
    images["group_id"] = images.index.to_series().groupby(group_ids).transform("first")
    overlap = images.index.isin(milk.isic_id) | images.index.isin([images.index[a] for a, _, _ in cross])
    conflicting = images.groupby("group_id").diagnosis_code.transform("nunique").gt(1)
    images["split"] = np.select(
        [images.group_id.isin(images.loc[overlap, "group_id"]), conflicting],
        ["excluded_milk_overlap", "excluded_conflicting_labels"],
        default="train",
    )
    manifest = images.loc[images.diagnosis_code.isin(["MEL", "NV"])].copy()
    manifest["target"] = manifest.diagnosis_code.eq("MEL").astype(int)
    eligible = manifest.loc[manifest.split.eq("train")]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = [held_out for _, held_out in splitter.split(eligible, eligible.target, eligible.group_id)]
    for name, rows in zip(("validation", "test"), folds[:2], strict=True):
        manifest.loc[eligible.index[rows], "split"] = name
    audit = pd.DataFrame(
        [(images.index[a], dermoscopy.isic_id.iloc[b], "MILK10k", score) for a, b, score in cross]
        + [(images.index[a], images.index[b], "ISIC2017", score) for a, b, score in within],
        columns=["image_id", "matched_image_id", "matched_source", "thumbnail_correlation"],
    ).round({"thumbnail_correlation": 5})
    return manifest, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[2] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_dir)))
    data_dir = parser.parse_args().data_dir.expanduser().resolve()
    manifest, audit = split_isic2017(data_dir)
    save_manifest(manifest, data_dir / "splits" / MANIFEST)
    save_manifest(audit, data_dir / "splits" / OVERLAP)
    print(pd.crosstab(manifest.diagnosis_code, manifest.split, margins=True).to_string())
    print(f"Image matches by source: {audit.matched_source.value_counts().to_dict()}")
    print(f"Manifest: {data_dir / 'splits' / MANIFEST}")


if __name__ == "__main__":
    main()
