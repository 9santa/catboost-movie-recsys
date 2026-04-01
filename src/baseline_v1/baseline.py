from __future__ import annotations

import pandas as pd

SCORE_COL = "weighted_popularity_score"


def add_popularity_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["score"] = result[SCORE_COL]
    return result
