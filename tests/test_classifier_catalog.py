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
                ("ISIC2017", None, "nv", None, "resized-nv.jpg", None, "NV"),
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
        milk = pd.DataFrame(
            {
                "lesion_id": ["lesion"],
                "diagnosis_code": ["MEL"],
                "split": ["train"],
                "validation_fold": [2],
                "group_id": ["group"],
                "clinical_path": ["old/clinic.jpg"],
                "dermoscopic_path": ["old/derm.jpg"],
            }
        )

        pairs = catalog.milk_lesions(milk)
        assignment_columns = milk.columns[:-2]
        pd.testing.assert_frame_equal(pairs.reindex(columns=assignment_columns), milk.reindex(columns=assignment_columns))
        self.assertEqual(pairs[["clinical_path", "dermoscopic_path"]].iloc[0].tolist(), ["resized-clinic.jpg", "resized-derm.jpg"])

    def test_milk_catalog_checks_other_diagnoses(self) -> None:
        rows = self.rows.drop(columns="diagnosis_code").assign(MEL=0, NV=[1, 0, 0, 0], BCC=[0, 1, 0, 1], BKL=0)
        lesions = pd.DataFrame(
            {"lesion_id": ["lesion"], "diagnosis_code": ["BCC"], "clinical_path": ["old/clinic.jpg"], "dermoscopic_path": ["old/derm.jpg"]}
        )
        rows.to_csv(self.catalog_path, index=False)
        catalog = load_catalog(self.catalog_path, self.data_dir)
        self.assertEqual(catalog.milk_lesions(lesions).at[0, "diagnosis_code"], "BCC")
        rows.loc[1, "BKL"] = 1
        rows.to_csv(self.catalog_path, index=False)
        drifted_catalog = load_catalog(self.catalog_path, self.data_dir)
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            drifted_catalog.milk_lesions(lesions)
