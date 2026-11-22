"""Map raw melanoma scores to the research demo's outcome bands."""

MODERATE_MELANOMA_SCORE = 0.50
HIGH_MELANOMA_SCORE = 0.90


def melanoma_score_band(score: float, specialist_cutoff: float) -> str:
    if score < specialist_cutoff:
        return "below"
    if score >= HIGH_MELANOMA_SCORE:
        return "high"
    if score >= MODERATE_MELANOMA_SCORE:
        return "moderate"
    return "low"
