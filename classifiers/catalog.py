"""Resolve model inputs from the image catalog using the frozen split assignments."""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

MILK_VIEWS = {"clinical: close-up": "clinical_path", "dermoscopic": "dermoscopic_path"}
CATALOG_COLUMNS = {"dataset", "lesion_id", "image_id", "image_type", "path", "age"}


@dataclass(frozen=True)
class ImageCatalog:
    rows: pd.DataFrame
    data_dir: Path
    sha256: str

    def _image_paths(self, images: pd.DataFrame, targets: pd.Series, description: str) -> list[str]:
        melanoma = targets.to_numpy()
        valid = images["MEL"].eq(melanoma) & images["NV"].eq(1 - melanoma)
        if "diagnosis_code" in images:
            valid &= images["diagnosis_code"].eq(targets.map({1: "MEL", 0: "NV"}).to_numpy())
        if not valid.fillna(False).all():
            raise ValueError(f"Catalog MEL/NV labels disagree with the frozen manifest for {description}")
        paths = images["path"].tolist()
        for path in paths:
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts or not (self.data_dir / relative).is_file():
                raise ValueError(f"Expected an existing path relative to --data-dir for {description}: {path}")
        return paths

    def milk_lesions(self, lesions: pd.DataFrame) -> pd.DataFrame:
        images = self.rows.loc[self.rows["dataset"].eq("MILK10k") & self.rows["lesion_id"].isin(lesions["lesion_id"])]
        if len(images) != len(MILK_VIEWS) * len(lesions):
            raise ValueError("Expected one clinical and one dermoscopic catalog row per MILK10k lesion in the frozen manifest")
        result = lesions.copy()
        for modality, column in MILK_VIEWS.items():
            view = images.loc[images["image_type"].eq(modality)].set_index("lesion_id")
            if not view.index.is_unique:
                raise ValueError(f"Expected one {modality} catalog row per MILK10k lesion")
            view = view.reindex(lesions["lesion_id"])
            expected_ids = lesions[column].map(lambda path: Path(path).stem)
            if not view["image_id"].eq(expected_ids.to_numpy()).fillna(False).all():
                raise ValueError(f"Catalog MILK10k {modality} image IDs disagree with the frozen manifest")
            result[column] = self._image_paths(view, lesions["target"], f"MILK10k {modality} images")
            if column == "clinical_path":
                result["age"] = view["age"].to_numpy()
        return result

    def isic_images(self, manifest: pd.DataFrame) -> pd.DataFrame:
        images = self.rows.loc[self.rows["dataset"].eq("ISIC2017")].set_index("image_id").reindex(manifest["image_id"])
        if not images["image_type"].eq("dermoscopic").fillna(False).all():
            raise ValueError("Expected one dermoscopic catalog row per ISIC image in the frozen manifest")
        return manifest.assign(image_path=self._image_paths(images, manifest["target"], "ISIC 2017 images"))


def load_catalog(path: Path | None, data_dir: Path, *, expected_sha256: str | None = None) -> ImageCatalog | None:
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("Pass --catalog with the image catalog used in the source run")
        return None

    content = (data_dir / path.expanduser()).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Image catalog differs from the version used in the source run")

    rows = pd.read_csv(BytesIO(content), dtype={key: "string" for key in ("dataset", "lesion_id", "image_id", "image_type", "path")})
    missing = CATALOG_COLUMNS.difference(rows.columns)
    if missing:
        raise ValueError(f"Image catalog is missing columns: {', '.join(sorted(missing))}")
    if not {"MEL", "NV"}.issubset(rows.columns):
        if "diagnosis_code" not in rows:
            raise ValueError("Expected MEL and NV flags or a diagnosis_code in the image catalog")
        rows = rows.assign(MEL=rows["diagnosis_code"].eq("MEL"), NV=rows["diagnosis_code"].eq("NV"))
    if rows[["dataset", "image_id", "path"]].isna().to_numpy().any() or rows.duplicated(["dataset", "image_id"]).any():
        raise ValueError("Expected complete catalog paths and unique image IDs within each dataset")
    if not rows["path"].is_unique:
        raise ValueError("Expected a distinct path per image in the catalog")
    return ImageCatalog(rows, data_dir, digest)
