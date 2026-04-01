from __future__ import annotations

import json
from pathlib import Path

from src.baseline_v1.baseline import add_popularity_score
from src.baseline_v1.prepare_data import make_baseline_dataset
from src.common.metrics import evaluate_ranking


def main() -> None:
    data_dir = Path("data/raw/ml-1m")
    report_dir = Path("reports/baseline_v1")
    report_dir.mkdir(parents=True, exist_ok=True)

    df = make_baseline_dataset(
        data_dir=data_dir,
        history_frac=0.8,
        positive_threshold=4.0,
        n_negative_candidates=100,
        min_votes=20,
        random_state=42,
    )

    scored_df = add_popularity_score(df)
    top_k = 10
    metrics = evaluate_ranking(
        df=scored_df,
        score_col="score",
        k=top_k,
    )

    print("Popularity baseline")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

    metrics_path = report_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    sample_predictions = []
    for user_id, group in scored_df.groupby("group_id", sort=False):
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
                    "score",
                    "rank",
                ]
            ].head(top_k)
        )

    sample_df = (
        __import__("pandas").concat(sample_predictions, ignore_index=True)
        if sample_predictions
        else scored_df.head(0)
    )
    sample_df.to_csv(report_dir / "sample_predictions.csv", index=False)


if __name__ == "__main__":
    main()
