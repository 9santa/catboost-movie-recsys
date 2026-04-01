from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

OCCUPATION_MAP = {
    0: "other_or_not_specified",
    1: "academic_educator",
    2: "artist",
    3: "clerical_admin",
    4: "college_grad_student",
    5: "customer_service",
    6: "doctor_health_care",
    7: "executive_managerial",
    8: "farmer",
    9: "homemaker",
    10: "k12_student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales_marketing",
    15: "scientist",
    16: "self_employed",
    17: "technician_engineer",
    18: "tradesman_craftsman",
    19: "unemployed",
    20: "writer",
}

AGE_MAP = {
    1: "under_18",
    18: "18_24",
    25: "25_34",
    35: "35_44",
    45: "45_49",
    50: "50_55",
    56: "56_plus",
}


def load_movielens_1m(
    data_dir: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает MovieLens 1M dataset.
    """
    data_dir = Path(data_dir)

    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",  # because engine 'c' doesn't support this separator
        encoding="latin-1",
    )

    users = pd.read_csv(
        data_dir / "users.dat",
        sep="::",
        names=["userId", "gender", "age_code", "occupation_code", "zip_code"],
        engine="python",
        encoding="latin-1",
    )

    movies = pd.read_csv(
        data_dir / "movies.dat",
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s", origin="unix")

    # map 'users' encoded features
    users["age_group"] = users["age_code"].map(AGE_MAP)
    users["occupation_name"] = users["occupation_code"].map(OCCUPATION_MAP)

    # in case of some unexpected values
    users["age_group"] = users["age_group"].fillna("unknown_age_group")
    users["occupation_name"] = users["occupation_name"].fillna("unknown_occupation")

    movies["genres_list"] = movies["genres"].str.split("|")
    movies["primary_genre"] = movies["genres_list"].str[0]
    movies["movie_year"] = (
        pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)")[0], errors="coerce"),
    )

    # Convert to strings, so CatBoost can easily consider them as categorical features
    users["gender"] = users["gender"].astype(str)
    users["age_code"] = users["age_code"].astype(str)
    users["age_group"] = users["age_group"].astype(str)
    users["occupation_code"] = users["occupation_code"].atype(str)
    users["occupation_name"] = users["occupation_name"].atype(str)
    movies["primary_genre"] = movies["primary_genre"].astype(str)
    movies["genres"] = movies["genres"].astype(str)
    movies["title"] = movies["title"].astype(str)

    return ratings, users, movies
