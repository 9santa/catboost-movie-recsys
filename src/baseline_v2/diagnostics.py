from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_v2.prepare_data import make_baseline_dataset


def build_global_target_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сравнивает распределения признаков у positive и negative примеров.
    """
    summary = (
        df.groupby("target")[["genre_preference_score", "popularity_score_norm"]]
        .agg(["mean", "median", "std", "min", "max"])
        .round(6)
    )
    return summary


def build_user_level_separation(
    df: pd.DataFrame, score_col: str
) -> tuple[pd.DataFrame, dict]:
    """
    Для каждого пользователя сравнивает средний score у positive и negative кандидатов.
    """
    grouped = (
        df.groupby(["userId", "target"])[score_col]
        .mean()
        .unstack()
        .rename(columns={0: "neg_mean", 1: "pos_mean"})
    )

    grouped = grouped.dropna(subset=["neg_mean", "pos_mean"]).copy()
    grouped["delta"] = grouped["pos_mean"] - grouped["neg_mean"]
    grouped["pos_gt_neg"] = grouped["delta"] > 0
    grouped["pos_eq_neg"] = np.isclose(grouped["delta"], 0.0)

    stats = {
        "score_col": score_col,
        "n_users_with_both_classes": int(len(grouped)),
        "share_users_pos_gt_neg": float(grouped["pos_gt_neg"].mean())
        if len(grouped)
        else 0.0,
        "share_users_pos_eq_neg": float(grouped["pos_eq_neg"].mean())
        if len(grouped)
        else 0.0,
        "mean_delta_pos_minus_neg": float(grouped["delta"].mean())
        if len(grouped)
        else 0.0,
        "median_delta_pos_minus_neg": float(grouped["delta"].median())
        if len(grouped)
        else 0.0,
    }

    return grouped.reset_index(), stats


def build_quantile_diagnostic(
    df: pd.DataFrame, score_col: str, n_bins: int = 10
) -> pd.DataFrame:
    """
    Делит объекты на бины по score и смотрит, как меняется доля позитивов.
    """
    tmp = df[[score_col, "target"]].copy()

    # qcut может падать на дубликатах, поэтому используем rank
    tmp["_ranked_score"] = tmp[score_col].rank(method="first")

    n_bins = min(n_bins, len(tmp))
    tmp["score_bin"] = pd.qcut(tmp["_ranked_score"], q=n_bins, labels=False) + 1

    bucket_stats = (
        tmp.groupby("score_bin")
        .agg(
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
            score_mean=(score_col, "mean"),
            positive_rate=("target", "mean"),
            n_samples=("target", "size"),
        )
        .reset_index()
        .round(6)
    )

    return bucket_stats


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

    # 1. Глобальное сравнение positive vs negative
    global_summary = build_global_target_summary(df)
    global_summary.to_csv(report_dir / "diagnostics_global_summary.csv")

    print("\n=== Global target summary ===")
    print(global_summary)

    # 2. User-level separation для genre_preference_score
    genre_user_df, genre_user_stats = build_user_level_separation(
        df=df,
        score_col="genre_preference_score",
    )
    genre_user_df.to_csv(report_dir / "diagnostics_user_level_genre.csv", index=False)

    print("\n=== User-level separation: genre_preference_score ===")
    for k, v in genre_user_stats.items():
        print(f"{k}: {v}")

    # 3. User-level separation для popularity_score_norm — для сравнения
    pop_user_df, pop_user_stats = build_user_level_separation(
        df=df,
        score_col="popularity_score_norm",
    )
    pop_user_df.to_csv(
        report_dir / "diagnostics_user_level_popularity.csv", index=False
    )

    print("\n=== User-level separation: popularity_score_norm ===")
    for k, v in pop_user_stats.items():
        print(f"{k}: {v}")

    # 4. Бины по genre_preference_score
    genre_bins = build_quantile_diagnostic(
        df=df,
        score_col="genre_preference_score",
        n_bins=10,
    )
    genre_bins.to_csv(report_dir / "diagnostics_genre_bins.csv", index=False)

    print("\n=== Quantile diagnostic: genre_preference_score ===")
    print(genre_bins)

    # 5. Бины по popularity_score_norm — для сравнения
    pop_bins = build_quantile_diagnostic(
        df=df,
        score_col="popularity_score_norm",
        n_bins=10,
    )
    pop_bins.to_csv(report_dir / "diagnostics_popularity_bins.csv", index=False)

    print("\n=== Quantile diagnostic: popularity_score_norm ===")
    print(pop_bins)

    summary = {
        "genre_user_stats": genre_user_stats,
        "popularity_user_stats": pop_user_stats,
    }

    with (report_dir / "diagnostics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
