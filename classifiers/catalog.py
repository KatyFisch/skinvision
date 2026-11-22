"""Resolve model inputs from the image catalog using the frozen split assignments."""

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import pandas as pd

MILK_VIEWS = {"clinical: close-up": "clinical_path", "dermoscopic": "dermoscopic_path"}
MILK_DIAGNOSES = ("AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC")
CATALOG_COLUMNS = {"dataset", "lesion_id", "image_id", "image_type", "path"}


@dataclass(frozen=True)
class ImageCatalog:
    rows: pd.DataFrame
    data_dir: Path
    sha256: str

    def _image_paths(self, images: pd.DataFrame, diagnoses: pd.Series, description: str) -> list[str]:
        expected = diagnoses.fillna("").to_numpy(dtype=str)
        valid = pd.Series(diagnoses.isin(MILK_DIAGNOSES).to_numpy(), index=images.index)
        flags = [code for code in MILK_DIAGNOSES if code in images]
        if "diagnosis_code" in images:
            valid &= images["diagnosis_code"].eq(expected)
        else:
            valid &= diagnoses.isin(flags).to_numpy()
        for code in flags:
            valid &= images[code].eq(expected == code)
        if not valid.fillna(False).all():
            raise ValueError(f"Catalog labels disagree with the frozen manifest for {description}")
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
        diagnoses = cast(pd.Series, lesions["diagnosis_code"])
        for modality, column in MILK_VIEWS.items():
            view = images.loc[images["image_type"].eq(modality)].set_index("lesion_id")
            if not view.index.is_unique:
                raise ValueError(f"Expected one {modality} catalog row per MILK10k lesion")
            view = view.reindex(lesions["lesion_id"])
            expected_ids = lesions[column].map(lambda path: Path(path).stem)
            if not view["image_id"].eq(expected_ids.to_numpy()).fillna(False).all():
                raise ValueError(f"Catalog MILK10k {modality} image IDs disagree with the frozen manifest")
            result[column] = self._image_paths(view, diagnoses, f"MILK10k {modality} images")
        return result


def load_catalog(path: Path, data_dir: Path) -> ImageCatalog:
    content = (data_dir / path.expanduser()).read_bytes()
    rows = pd.read_csv(BytesIO(content), dtype={key: "string" for key in ("dataset", "lesion_id", "image_id", "image_type", "path")})
    missing = CATALOG_COLUMNS.difference(rows.columns)
    if missing:
        raise ValueError(f"Image catalog is missing columns: {', '.join(sorted(missing))}")
    if not {"MEL", "NV"}.issubset(rows.columns):
        if "diagnosis_code" not in rows:
            raise ValueError("Expected MEL and NV flags or a diagnosis_code in the image catalog")
        for code in {"MEL", "NV"}.difference(rows.columns):
            rows[code] = rows["diagnosis_code"].eq(code)
    if rows[["dataset", "image_id", "path"]].isna().to_numpy().any() or rows.duplicated(["dataset", "image_id"]).any():
        raise ValueError("Expected complete catalog paths and unique image IDs within each dataset")
    if not rows["path"].is_unique:
        raise ValueError("Expected a distinct path per image in the catalog")
    return ImageCatalog(rows, data_dir, hashlib.sha256(content).hexdigest())
