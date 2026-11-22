"""Run the saved padded-crop melanoma-versus-nevus classifier."""

import json
import math
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import torch
from PIL import Image

from classifiers import dermfm_zero
from classifiers.lesion_crop import CROP_MARGIN_FRACTION, SKIN_MARGIN_PX


class PaddedCropClassifier:
    def __init__(self, run_dir: Path, data_dir: Path, device: torch.device) -> None:
        run_dir = run_dir.expanduser().resolve()
        config = json.loads((run_dir / "metrics.json").read_text())
        expected_crop = {"margin_fraction_of_longest_side": CROP_MARGIN_FRACTION, "minimum_margin_px": SKIN_MARGIN_PX}
        if (
            config.get("context_transforms", {}).get("padded_crop") != expected_crop
            or config.get("empty_mask_policy") != "original_photo"
            or config.get("encoder", {}).get("sha256") != dermfm_zero.WEIGHTS_SHA256
        ):
            raise ValueError("Expected a DermFM-Zero-Open padded-crop run with the original-photo fallback")

        self.cutoff = float(config["methods"]["padded_crop"]["cutoff"])
        self.heads = joblib.load(run_dir / "heads.joblib")["padded_crop"]
        if not 0 <= self.cutoff <= 1 or len(self.heads) != 3:
            raise ValueError("Expected a valid cutoff and three fitted padded-crop heads")
        self.device = device
        self.preprocess = dermfm_zero.clinical_preprocess()
        self.encoder = dermfm_zero.load_clinical_encoder(data_dir, device)

    @torch.inference_mode()
    def predict(self, photo: Image.Image) -> dict[str, str | float | None]:
        tensor = cast(torch.Tensor, self.preprocess(photo))
        image = tensor.unsqueeze(0).to(device=self.device, dtype=dermfm_zero.clinical_dtype(self.device))
        features = self.encoder(image).float().cpu().numpy()
        score = float(np.mean([head.predict_proba(features)[0, 1] for head in self.heads]))
        if not math.isfinite(score):
            raise RuntimeError("The padded-crop classifier returned an invalid melanoma score")
        return {"research_prediction": "MEL" if score >= self.cutoff else "NV", "route": "MEL_OR_NV", "melanoma_score": score}
