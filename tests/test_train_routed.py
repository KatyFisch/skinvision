from unittest import TestCase

import numpy as np
import pandas as pd

from classifiers.train_routed import summarize_router


class RouterMetricsTest(TestCase):
    def test_cutoff_uses_melanoma_recall_and_counts_other_routes_separately(self) -> None:
        rows = pd.DataFrame({"diagnosis_code": ["MEL", "MEL", "NV", "NV", "BCC", "DF"], "target": [1, 1, 1, 1, 0, 0]})
        scores = np.array([0.9, 0.4, 0.8, 0.2, 0.6, 0.1])

        metrics = summarize_router(rows, scores, melanoma_sensitivity=1.0)

        self.assertEqual(metrics["threshold"], 0.4)
        self.assertEqual(metrics["melanoma_routing_sensitivity"], 1)
        self.assertEqual(metrics["by_diagnosis"]["NV"], {"lesions": 2, "sent_to_specialist": 1})
        self.assertEqual(metrics["by_diagnosis"]["BCC"], {"lesions": 1, "sent_to_specialist": 1})
        self.assertEqual(metrics["other_routing_specificity"], 0.5)
