from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.baseline_v2.baseline import add_baseline_v2_scores
from src.baseline_v2.prepare_data import make_baseline_dataset
from src.common.metrics import evaluate_ranking


def main() -> None:
    data_dir = Path("data/raw/ml-1m")
    report_dir = Path("reports/baseline_v2")
    report_dir.mkdir(parents=True, exist_ok=True)

    df = make_baseline_dataset(
        data_dir=data_dir,
        history_frac=0.8,
        positive_threshold=4.0,
        n_negative_candidates=100,
        min_votes=20,
        random_state=42,
    )

    scored_df = add_baseline_v2_scores(
        df=df,
        alpha=0.0,
        beta=1.0,
    )

    metrics = evaluate_ranking(
        df=scored_df,
        score_col="score",
        k=10,
    )

    print("Baseline v2: genre preference + popularity")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    sample_predictions = []
    for _, group in scored_df.groupby("group_id", sort=False):
        group = group.sort_values("score", ascending=False).copy()
        group["rank"] = range(1, len(group) + 1)

        sample_predictions.append(
            group[
                [
                    "userId",
                    "movieId",
                    "title",
                    "genres",
                    "target",
                    "genre_preference_score",
                    "popularity_score_norm",
                    "score",
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
