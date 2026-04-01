from __future__ import annotations

import pandas as pd


def add_baseline_v2_scores(
    df: pd.DataFrame,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> pd.DataFrame:
    """
    final_score = alpha * genre_preference_score + beta * popularity_score_norm
    """
    result = df.copy()

    result["score"] = (
        alpha * result["genre_preference_score"] + beta * result["popularity_score_norm"]
    )

    return result
