from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.load_data import load_movielens_1m
from src.common.split_data import time_split_per_user, build_candidates


def build_movie_stats(history: pd.DataFrame, min_votes: int = 20) -> pd.DataFrame:
    """
    Считает movie-level stats только по history.
    """
    movie_stats = (
        history.groupby("movieId")
        .agg(
            movie_mean_rating=("rating", "mean"),
            movie_rating_count=("rating", "size"),
            movie_std_rating=("rating", "std"),
        )
        .reset_index()
        .fillna({"movie_std_rating": 0.0})
    )

    global_mean_rating = history["rating"].mean()

    v = movie_stats["movie_rating_count"]
    R = movie_stats["movie_mean_rating"]
    m = min_votes
    C = global_mean_rating
    """
    Логика формулы:
        - если у фильма мало оценок, доверяем ему меньше;
        - если много - больше верим его среднему рейтингу;
    """
    movie_stats["weighted_popularity_score"] = (v / (v + m)) * R + (m / (v + m)) * C

    return movie_stats


def make_baseline_dataset(
    data_dir: str | Path,
    history_frac: float = 0.8,
    positive_threshold: float = 4.0,
    n_negative_candidates: int = 100,
    min_votes: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    ratings, users, movies = load_movielens_1m(data_dir)

    history, future = time_split_per_user(
        ratings=ratings, history_frac=history_frac, min_ratings_per_user=20
    )

    candidates = build_candidates(
        history=history,
        future=future,
        movies=movies,
        n_negative_candidates=n_negative_candidates,
        positive_threshold=positive_threshold,
        min_positive_future=1,
        random_state=random_state,
    )

    movie_stats = build_movie_stats(history=history, min_votes=min_votes)

    df = candidates.merge(movie_stats, on="movieId", how="left")
    df = df.merge(
        movies[["movieId", "title", "genres", "primary_genre", "movie_year"]],
        on="movieId",
        how="left",
    )

    global_mean_rating = history["rating"].mean()

    df = df.fillna(
        {
            "movie_mean_rating": global_mean_rating,
            "movie_rating_count": 0,
            "movie_std_rating": 0.0,
            "weighted_popularity_score": global_mean_rating,
            "title": "unknown_title",
            "genres": "unknown_genres",
            "primary_genre": "unknown_genre",
            "movie_year": -1,
        }
    )

    df["movie_year"] = df["movie_year"].astype(int)

    return df.sort_values(
        ["group_id", "weighted_popularity_score"], ascending=[True, False]
    )
