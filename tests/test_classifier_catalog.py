from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from classifiers.catalog import load_catalog


class ImageCatalogTest(TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.data_dir = Path(directory.name)
        self.catalog_path = self.data_dir / "final_data.csv"
        self.rows = pd.DataFrame(
            [
                ("ISIC2017", None, "nv", "dermoscopic", "resized-nv.jpg", None, "NV"),
                ("MILK10k", "lesion", "derm", "dermoscopic", "resized-derm.jpg", 50, "MEL"),
                ("ISIC2017", None, "sk", "dermoscopic", "missing-sk.jpg", None, "SK"),
                ("MILK10k", "lesion", "clinic", "clinical: close-up", "resized-clinic.jpg", 45, "MEL"),
            ],
            columns=["dataset", "lesion_id", "image_id", "image_type", "path", "age", "diagnosis_code"],
        )
        for path in ("resized-nv.jpg", "resized-derm.jpg", "resized-clinic.jpg"):
            (self.data_dir / path).touch()

    def test_joins_keep_frozen_assignments_and_resolve_catalog_paths(self) -> None:
        self.rows.to_csv(self.catalog_path, index=False)
        catalog = load_catalog(Path(self.catalog_path.name), self.data_dir)
        assert catalog is not None
        milk = pd.DataFrame(
            {
                "lesion_id": ["lesion"],
                "target": [1],
                "split": ["train"],
                "validation_fold": [2],
                "group_id": ["group"],
                "clinical_path": ["old/clinic.jpg"],
                "dermoscopic_path": ["old/derm.jpg"],
            }
        )
        isic = pd.DataFrame({"image_id": ["nv"], "target": [0], "split": ["validation"], "image_path": ["old/nv.jpg"]})

        pairs = catalog.milk_lesions(milk)
        pd.testing.assert_frame_equal(pairs[milk.columns[:-2]], milk[milk.columns[:-2]])
        self.assertEqual(
            pairs[["clinical_path", "dermoscopic_path", "age"]].iloc[0].tolist(), ["resized-clinic.jpg", "resized-derm.jpg", 45]
        )
        views = catalog.isic_images(isic)
        pd.testing.assert_frame_equal(views[isic.columns[:-1]], isic[isic.columns[:-1]])
        self.assertEqual(views.at[0, "image_path"], "resized-nv.jpg")

    def test_rejects_label_drift_and_catalog_changes(self) -> None:
        self.rows = self.rows.drop(columns="diagnosis_code").assign(MEL=[1, 1, 0, 1], NV=[0, 0, 0, 0])
        self.rows.to_csv(self.catalog_path, index=False)
        catalog = load_catalog(self.catalog_path, self.data_dir)
        assert catalog is not None

        with self.assertRaisesRegex(ValueError, "labels disagree"):
            catalog.isic_images(pd.DataFrame({"image_id": ["nv"], "target": [0]}))
        with self.assertRaisesRegex(ValueError, "Pass --catalog"):
            load_catalog(None, self.data_dir, expected_sha256=catalog.sha256)
        self.rows.loc[0, ["MEL", "NV"]] = [0, 1]
        self.rows.to_csv(self.catalog_path, index=False)
        with self.assertRaisesRegex(ValueError, "differs from the version"):
            load_catalog(self.catalog_path, self.data_dir, expected_sha256=catalog.sha256)
