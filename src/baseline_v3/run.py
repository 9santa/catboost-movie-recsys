from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.baseline_v3.model import (
    add_pointwise_scores,
    fit_pointwise_model,
    get_feature_importance_df,
    train_valid_split_by_user,
)
from src.baseline_v3.prepare_data import get_feature_columns, make_pointwise_dataset
from src.common.metrics import evaluate_ranking


def main() -> None:
    data_dir = Path("data/raw/ml-1m")
    report_dir = Path("reports/baseline_v3")
    report_dir.mkdir(parents=True, exist_ok=True)

    df = make_pointwise_dataset(
        data_dir=data_dir,
        history_frac=0.8,
        positive_threshold=4.0,
        n_negative_candidates=100,
        min_votes=20,
        random_state=42,
    )

    feature_cols, cat_features, num_features = get_feature_columns()

    train_df, valid_df = train_valid_split_by_user(
        df=df,
        valid_size=0.2,
        random_state=42,
    )

    print(f"Train shape: {train_df.shape}")
    print(f"Valid shape: {valid_df.shape}")
    print(f"N features: {len(feature_cols)}")
    print(f"Categorical features: {cat_features}")

    model = fit_pointwise_model(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        cat_features=cat_features,
        random_state=42,
    )

    scored_valid_df = add_pointwise_scores(
        model=model,
        df=valid_df,
        feature_cols=feature_cols,
    )

    metrics = evaluate_ranking(
        df=scored_valid_df,
        score_col="score",
        k=10,
    )

    print("\nBaseline v3: pointwise CatBoostClassifier")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    fi_df = get_feature_importance_df(model, feature_cols)
    fi_df.to_csv(report_dir / "feature_importance.csv", index=False)

    sample_predictions = []
    for _, group in scored_valid_df.groupby("group_id", sort=False):
        group = group.sort_values("score", ascending=False).copy()
        group["rank"] = range(1, len(group) + 1)

        sample_predictions.append(
            group[
                [
                    "userId",
                    "movieId",
                    "target",
                    "score",
                    "genres",
                    "primary_genre",
                    "weighted_popularity_score",
                    "genre_preference_score",
                    "rank",
                ]
            ].head(10)
        )

        if len(sample_predictions) >= 20:
            break

    pd.concat(sample_predictions, ignore_index=True).to_csv(
        report_dir / "sample_predictions.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
