from __future__ import annotations

import numpy as np
import pandas as pd


def dcg_at_k(targets: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    rel = targets[order]

    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    gains = (2**rel - 1) * discounts
    return float(gains.sum())


def ndcg_at_k(targets: np.ndarray, scores: np.ndarray, k: int) -> float:
    actual_dcg = dcg_at_k(targets, scores, k)

    ideal_order = np.argsort(targets)[::-1][:k]
    ideal_rel = targets[ideal_order]

    discounts = 1.0 / np.log2(np.arange(2, len(ideal_rel) + 2))
    ideal_dcg = float(((2**ideal_rel - 1) * discounts).sum())

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def recall_at_k(targets: np.ndarray, scores: np.ndarray, k: int) -> float:
    n_positives = int(targets.sum())
    if n_positives == 0:
        return 0.0

    order = np.argsort(scores)[::-1][:k]
    rel = targets[order]
    return float(rel.sum() / n_positives)


def mrr_at_k(targets: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    rel = targets[order]

    positive_positions = np.where(rel > 0)[0]
    if len(positive_positions) == 0:
        return 0.0

    return float(1.0 / (positive_positions[0] + 1))


def evaluate_ranking(df: pd.DataFrame, score_col: str, k: int = 10) -> dict:
    ndcgs = []
    recalls = []
    mrrs = []

    for _, group in df.groupby("group_id", sort=False):
        targets = group["target"].to_numpy()
        scores = group[score_col].to_numpy()

        if targets.sum() == 0:
            continue

        ndcgs.append(ndcg_at_k(targets, scores, k))
        recalls.append(recall_at_k(targets, scores, k))
        mrrs.append(mrr_at_k(targets, scores, k))

    return {
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"MRR@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
    }
