from __future__ import annotations

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier, Pool


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


def fit_pointwise_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    random_state: int = 42,
) -> CatBoostClassifier:
    train_pool = Pool(
        data=train_df[feature_cols],
        label=train_df["target"],
        cat_features=cat_features,
    )

    valid_pool = Pool(
        data=valid_df[feature_cols],
        label=valid_df["target"],
        cat_features=cat_features,
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
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


def add_pointwise_scores(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    result = df.copy()
    result["score"] = model.predict_proba(result[feature_cols])[:, 1]
    return result


def get_feature_importance_df(
    model: CatBoostClassifier,
    feature_cols: list[str],
) -> pd.DataFrame:
    importances = model.get_feature_importance()
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
