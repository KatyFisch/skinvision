from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

import pandas as pd

from classifiers.catalog import MILK_DIAGNOSES, ImageCatalog, load_catalog
from classifiers.eval_split.create_splits import FULL_MANIFEST_NAME, MANIFEST_NAME, filter_mel_nv
from classifiers.eval_split.prepare_routed_tasks import LESION_COLUMNS, OTHER_DIAGNOSIS_TARGETS, prepare_routed_tasks


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

    def test_milk_catalog_checks_other_diagnoses_when_router_targets_are_present(self) -> None:
        rows = self.rows.drop(columns="diagnosis_code").assign(MEL=0, NV=[1, 0, 0, 0], BCC=[0, 1, 0, 1], BKL=0)
        lesions = pd.DataFrame(
            {
                "lesion_id": ["lesion"],
                "diagnosis_code": ["BCC"],
                "target": [0],
                "clinical_path": ["old/clinic.jpg"],
                "dermoscopic_path": ["old/derm.jpg"],
            }
        )
        rows.to_csv(self.catalog_path, index=False)
        catalog = load_catalog(self.catalog_path, self.data_dir)
        assert catalog is not None
        self.assertEqual(catalog.milk_lesions(lesions).at[0, "diagnosis_code"], "BCC")
        rows.loc[1, "BKL"] = 1
        rows.to_csv(self.catalog_path, index=False)
        drifted_catalog = load_catalog(self.catalog_path, self.data_dir)
        assert drifted_catalog is not None
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            drifted_catalog.milk_lesions(lesions)


class RoutedTaskTest(TestCase):
    def test_targets_inherit_parent_assignments_and_require_the_specialist_split(self) -> None:
        with TemporaryDirectory() as directory:
            data_dir = Path(directory)
            splits = data_dir / "splits"
            splits.mkdir()
            rows = []
            for index, code in enumerate(MILK_DIAGNOSES):
                held_out = code in {"NV", "VASC"}
                rows.append(
                    (
                        code,
                        code,
                        f"{code}-clinical.jpg",
                        f"{code}-derm.jpg",
                        code,
                        "test" if held_out else "train",
                        None if held_out else index % 3 + 1,
                    )
                )
            full = pd.DataFrame(rows, columns=LESION_COLUMNS).astype({"validation_fold": "Int64"})
            full.to_csv(splits / FULL_MANIFEST_NAME, index=False)
            specialist = filter_mel_nv(full)
            specialist.to_csv(splits / MANIFEST_NAME, index=False)
            catalog = Mock(spec=ImageCatalog)
            catalog.milk_lesions.side_effect = pd.DataFrame.copy
            router, other = prepare_routed_tasks(data_dir, catalog)

            pd.testing.assert_frame_equal(router[full.columns[1:]], full.set_index("lesion_id"))
            self.assertEqual(router.loc[["MEL", "NV", "BCC"], "target"].tolist(), [1, 1, 0])
            self.assertEqual(other["target"].to_dict(), OTHER_DIAGNOSIS_TARGETS)
            self.assertEqual(other.at["VASC", "split"], "test")

            specialist.loc[specialist["diagnosis_code"].eq("MEL"), "validation_fold"] = 1
            specialist.to_csv(splits / MANIFEST_NAME, index=False)
            with self.assertRaisesRegex(ValueError, "disagrees with the frozen MEL-vs-NV manifest"):
                prepare_routed_tasks(data_dir, catalog)
