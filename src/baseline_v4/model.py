from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


def train_valid_split_by_user(
    df: pd.DataFrame,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_users = df["userId"].drop_duplicates().to_numpy()

    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_users)

    n_valid = int(len(unique_users) * valid_size)
    n_valid = max(1, n_valid)

    valid_users = set(unique_users[:n_valid])

    valid_df = df[df["userId"].isin(valid_users)].copy()
    train_df = df[~df["userId"].isin(valid_users)].copy()

    return train_df, valid_df


def sort_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Для CatBoost ranking объекты с одинаковым group_id должны идти подряд.
    """
    return df.sort_values(["group_id", "movieId"]).reset_index(drop=True)


def fit_ranker_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    random_state: int = 42,
) -> CatBoostRanker:
    train_pool = make_ranker_pool(train_df, feature_cols, cat_features)
    valid_pool = make_ranker_pool(valid_df, feature_cols, cat_features)

    model = CatBoostRanker(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        custom_metric=["MRR:top=10", "RecallAt:top=10"],
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=random_state,
        verbose=100,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=50,
    )

    return model


def add_ranker_scores(
    model: CatBoostRanker,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    result = sort_by_group(df.copy())
    result["score"] = model.predict(result[feature_cols])
    return result


def make_ranker_pool(
    df: pd.DataFrame, feature_cols: list[str], cat_features: list[str]
) -> Pool:
    df = sort_by_group(df)

    return Pool(
        data=df[feature_cols],
        label=df["target"],
        group_id=df["group_id"],
        cat_features=cat_features,
    )


def get_feature_importance_df(
    model: CatBoostRanker,
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
) -> pd.DataFrame:
    pool = make_ranker_pool(df, feature_cols, cat_features)
    importances = model.get_feature_importance(data=pool)
    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
