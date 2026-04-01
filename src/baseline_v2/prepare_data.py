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

    # Min-Max normalization to [0..1]
    pop_min = movie_stats["weighted_popularity_score"].min()
    pop_max = movie_stats["weighted_popularity_score"].max()

    if pop_max > pop_min:
        movie_stats["popularity_score_norm"] = (
            movie_stats["weighted_popularity_score"] - pop_min
        ) / (pop_max - pop_min)
    else:
        movie_stats["popularity_score_norm"] = 0.0

    return movie_stats


def build_user_global_preferences(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    history["liked"] = (history["rating"] >= 4).astype(int)

    user_prefs = (
        history.groupby("userId")
        .agg(
            user_mean_rating=("rating", "mean"),
            user_like_rate=("liked", "mean"),
            user_rating_count=("rating", "size"),
        )
        .reset_index()
    )

    return user_prefs


def build_user_genre_preferences(
    history: pd.DataFrame, movies: pd.DataFrame
) -> pd.DataFrame:
    hist_genres = history.merge(
        movies[["movieId", "genres_list"]],
        on="movieId",
        how="left",
    ).explode("genres_list")

    hist_genres = hist_genres.rename(columns={"genres_list": "genre"})
    hist_genres["liked"] = (hist_genres["rating"] >= 4).astype(int)

    user_genre_prefs = (
        hist_genres.groupby(["userId", "genre"])
        .agg(
            user_genre_like_rate=("liked", "mean"),
            user_genre_mean_rating=("rating", "mean"),
            user_genre_support=("rating", "size"),
        )
        .reset_index()
    )

    return user_genre_prefs


def build_genre_features(
    candidates: pd.DataFrame,
    movies: pd.DataFrame,
    user_genre_prefs: pd.DataFrame,
    user_global_prefs: pd.DataFrame,
) -> pd.DataFrame:
    df = candidates.merge(
        movies[
            ["movieId", "title", "genres", "genres_list", "primary_genre", "movie_year"]
        ],
        on="movieId",
        how="left",
    )

    cand_genres = df[["userId", "movieId", "genres_list"]].explode("genres_list")
    cand_genres = cand_genres.rename(columns={"genres_list": "genre"})

    cand_genres = cand_genres.merge(
        user_genre_prefs,
        on=["userId", "genre"],
        how="left",
    )

    cand_genres = cand_genres.merge(
        user_global_prefs[["userId", "user_like_rate", "user_mean_rating"]],
        on="userId",
        how="left",
    )

    # fallback:
    # если пользователь не встречался с жанром, используем его общий like rate / mean rating
    cand_genres["genre_like_signal"] = cand_genres["user_genre_like_rate"].fillna(
        cand_genres["user_like_rate"]
    )
    cand_genres["genre_rating_signal"] = cand_genres["user_genre_mean_rating"].fillna(
        cand_genres["user_mean_rating"]
    )
    cand_genres["genre_support_signal"] = cand_genres["user_genre_support"].fillna(0)

    genre_features = (
        cand_genres.groupby(["userId", "movieId"])
        .agg(
            genre_preference_score=("genre_like_signal", "mean"),
            genre_mean_rating_score=("genre_rating_signal", "mean"),
            genre_support_sum=("genre_support_signal", "sum"),
        )
        .reset_index()
    )

    return df.drop(columns=["genres_list"]).merge(
        genre_features,
        on=["userId", "movieId"],
        how="left",
    )


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
    user_global_prefs = build_user_global_preferences(history=history)
    user_genre_prefs = build_user_genre_preferences(history=history, movies=movies)

    df = candidates.merge(movie_stats, on="movieId", how="left")
    df = build_genre_features(
        candidates=df,
        movies=movies,
        user_genre_prefs=user_genre_prefs,
        user_global_prefs=user_global_prefs,
    )

    global_mean_rating = history["rating"].mean()

    df = df.fillna(
        {
            "movie_mean_rating": global_mean_rating,
            "movie_rating_count": 0,
            "movie_std_rating": 0.0,
            "weighted_popularity_score": global_mean_rating,
            "popularity_score_norm": 0.0,
            "genre_preference_score": 0.0,
            "genre_mean_rating_score": global_mean_rating,
            "genre_support_sum": 0.0,
            "title": "unknown_title",
            "genres": "unknown_genres",
            "primary_genre": "unknown_genre",
            "movie_year": -1,
        }
    )

    df["movie_year"] = df["movie_year"].astype(int)

    return df.reset_index(drop=True)
