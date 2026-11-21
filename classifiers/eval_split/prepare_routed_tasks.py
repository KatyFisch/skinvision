"""Prepare catalog-backed manifests for the MEL-or-NV router and other-diagnosis classifier."""

import argparse
import os
from pathlib import Path

import pandas as pd

from classifiers.catalog import MEL_NV_DIAGNOSES, MILK_DIAGNOSES, ImageCatalog, load_catalog
from classifiers.eval_split.create_splits import FULL_MANIFEST_NAME, MANIFEST_NAME, filter_mel_nv, save_manifest

ROUTER_MANIFEST = "milk10k-router-v1.csv"
OTHER_DIAGNOSES_MANIFEST = "milk10k-other-diagnoses-v1.csv"
OTHER_DIAGNOSIS_TARGETS = {
    code: index for index, code in enumerate(code for code in MILK_DIAGNOSES if code not in MEL_NV_DIAGNOSES.values())
}
LESION_COLUMNS = ["lesion_id", "diagnosis_code", "clinical_path", "dermoscopic_path", "group_id", "split", "validation_fold"]


def prepare_routed_tasks(data_dir: Path, catalog: ImageCatalog) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = pd.read_csv(data_dir / "splits" / FULL_MANIFEST_NAME, dtype={"validation_fold": "Int64"})[LESION_COLUMNS]
    full = full.sort_values("lesion_id").reset_index(drop=True)
    if full["lesion_id"].isna().any() or not full["lesion_id"].is_unique or set(full["diagnosis_code"]) != set(MILK_DIAGNOSES):
        raise ValueError("Expected unique MILK10k lesions and all 11 diagnoses in the parent manifest")
    if set(full["split"]) != {"train", "test"}:
        raise ValueError("Expected development and test rows in the parent manifest")
    development = full["split"].eq("train")
    if set(full.loc[development, "validation_fold"]) != {1, 2, 3} or full.loc[~development, "validation_fold"].notna().any():
        raise ValueError("Expected three development folds and no validation folds on test rows")
    groups = full[["group_id", "split", "validation_fold"]].drop_duplicates()
    if groups["group_id"].isna().any() or not groups["group_id"].is_unique:
        raise ValueError("Every MILK10k group must stay in one split or development fold")

    mel_nv = pd.read_csv(data_dir / "splits" / MANIFEST_NAME, dtype={"validation_fold": "Int64"})
    expected = filter_mel_nv(full).reset_index(drop=True)
    if not expected.equals(mel_nv[expected.columns].sort_values("lesion_id").reset_index(drop=True)):
        raise ValueError("The parent manifest disagrees with the frozen MEL-vs-NV manifest")

    lesions = catalog.milk_lesions(full)[LESION_COLUMNS]
    specialist = lesions["diagnosis_code"].isin(MEL_NV_DIAGNOSES.values())
    router = lesions.assign(target_name=specialist.map({True: "MEL_OR_NV", False: "OTHER"}), target=specialist.astype(int))
    other = lesions.loc[~specialist].assign(
        target_name=lambda rows: rows["diagnosis_code"], target=lambda rows: rows["diagnosis_code"].map(OTHER_DIAGNOSIS_TARGETS)
    )
    return router.set_index("lesion_id"), other.set_index("lesion_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[2] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_dir)))
    parser.add_argument("--catalog", type=Path, default=Path("final_data.csv"), help="Image catalog CSV relative to --data-dir or absolute")
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    catalog = load_catalog(args.catalog, data_dir)
    assert catalog is not None
    router, other = prepare_routed_tasks(data_dir, catalog)
    print(f"Catalog SHA-256: {catalog.sha256}")
    for name, manifest in ((ROUTER_MANIFEST, router), (OTHER_DIAGNOSES_MANIFEST, other)):
        path = data_dir / "splits" / name
        save_manifest(manifest, path)
        partition = manifest["validation_fold"].astype("string").fillna("test")
        print(f"Manifest: {path}")
        print(pd.crosstab(manifest["target_name"], partition, margins=True).to_string())


if __name__ == "__main__":
    main()
