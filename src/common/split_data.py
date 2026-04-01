from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def time_split_per_user(
    ratings: pd.DataFrame,
    history_frac: float = 0.8,
    min_ratings_per_user: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит рейтинги каждого пользователя по времени:
    первые history_frac -> history
    последние (history_frac) -> future
    """
    ratings = ratings.sort_values(["userId", "timestamp"]).copy()

    hist_parts = []
    fut_parts = []

    for user_id, g in ratings.groupby("userId", sort=False):
        if len(g) < min_ratings_per_user:
            continue

        cutoff_idx = max(1, int(len(g) * history_frac))
        # leave atleast 1 object for the future
        cutoff_idx = min(cutoff_idx, len(g) - 1)

        hist_parts.append(g.iloc[:cutoff_idx].copy())
        fut_parts.append(g.iloc[cutoff_idx:].copy())

    history = pd.concat(hist_parts, ignore_index=True)
    future = pd.concat(fut_parts, ignore_index=True)

    return history, future


def build_candidates(
    history: pd.DataFrame,
    future: pd.DataFrame,
    movies: pd.DataFrame,
    n_negative_candidates: int = 100,
    positive_threshold: float = 4.0,
    min_positive_future: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Собирает candidate set для каждого пользователя.
    positive:
        - фильмы из future, которые пользователь оценил на >= positive_threshold
    negative:
        - случайные фильмы, которых пользователь не видел в history
        и которые не являются positive в future
    """
    rng = np.random.default_rng(random_state)

    all_movie_ids = set(movies["movieId"].unique())

    # Builds a dictionary (userId: set(movieId), movieId - фильмы которые пользователь уже оценил)
    seen_history = history.groupby("userId")["movieId"].apply(set).to_dict()
    positive_future = (
        future[future["rating"] >= positive_threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    rows = []

    for user_id, seen_movies in seen_history.items():
        positive_movies = positive_future.get(user_id, set())

        if len(positive_movies) < min_positive_future:
            continue

        negative_pool = list(all_movie_ids - seen_movies - positive_movies)
        if len(negative_pool) == 0:
            continue

        n_neg = min(n_negative_candidates, len(negative_pool))
        sampled_negatives = rng.choice(
            negative_pool, size=n_neg, replace=False
        ).tolist()

        for movie_id in positive_movies:
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "target": 1,
                    "group_id": user_id,
                }
            )

        for movie_id in sampled_negatives:
            rows.append(
                {
                    "userId": user_id,
                    "movieId": movie_id,
                    "target": 0,
                    "group_id": user_id,
                }
            )

    candidates = pd.DataFrame(rows)

    if candidates.empty:
        raise ValueError("Candidates dataframe is empty.")

    return candidates
