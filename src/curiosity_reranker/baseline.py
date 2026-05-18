from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from curiosity_reranker.features import normalize_score
from curiosity_reranker.schema import UserProfile


@dataclass(frozen=True)
class MFModel:
    global_mean: float
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray
    user_to_idx: dict[int, int]
    item_to_idx: dict[int, int]
    idx_to_item: dict[int, int]


def leave_one_out_split(
    ratings: pd.DataFrame,
    min_interactions: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_users = ratings.groupby("userId").size()
    eligible_users = eligible_users[eligible_users >= min_interactions].index
    filtered = ratings[ratings["userId"].isin(eligible_users)].copy()
    filtered = filtered.sort_values(["userId", "timestamp"])

    test_idx = filtered.groupby("userId").tail(1).index
    test = filtered.loc[test_idx].copy()
    train = filtered.drop(test_idx).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_matrix_factorization(
    train_ratings: pd.DataFrame,
    n_factors: int = 32,
    epochs: int = 8,
    learning_rate: float = 0.02,
    regularization: float = 0.05,
    seed: int = 42,
) -> MFModel:
    rng = np.random.default_rng(seed)
    users = sorted(train_ratings["userId"].unique().tolist())
    items = sorted(train_ratings["movieId"].unique().tolist())
    user_to_idx = {int(user_id): idx for idx, user_id in enumerate(users)}
    item_to_idx = {int(item_id): idx for idx, item_id in enumerate(items)}
    idx_to_item = {idx: int(item_id) for item_id, idx in item_to_idx.items()}

    user_factors = rng.normal(0, 0.1, size=(len(users), n_factors))
    item_factors = rng.normal(0, 0.1, size=(len(items), n_factors))
    user_bias = np.zeros(len(users))
    item_bias = np.zeros(len(items))
    global_mean = float(train_ratings["rating"].mean())

    training_rows = [
        (user_to_idx[int(row.userId)], item_to_idx[int(row.movieId)], float(row.rating))
        for row in train_ratings.itertuples(index=False)
    ]

    for _ in range(epochs):
        rng.shuffle(training_rows)
        for user_idx, item_idx, rating in training_rows:
            pred = (
                global_mean
                + user_bias[user_idx]
                + item_bias[item_idx]
                + float(user_factors[user_idx] @ item_factors[item_idx])
            )
            error = rating - pred

            user_bias[user_idx] += learning_rate * (error - regularization * user_bias[user_idx])
            item_bias[item_idx] += learning_rate * (error - regularization * item_bias[item_idx])

            user_vec = user_factors[user_idx].copy()
            item_vec = item_factors[item_idx].copy()
            user_factors[user_idx] += learning_rate * (
                error * item_vec - regularization * user_vec
            )
            item_factors[item_idx] += learning_rate * (
                error * user_vec - regularization * item_vec
            )

    return MFModel(
        global_mean=global_mean,
        user_factors=user_factors,
        item_factors=item_factors,
        user_bias=user_bias,
        item_bias=item_bias,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        idx_to_item=idx_to_item,
    )


def predict_rating(model: MFModel, user_id: int, item_id: int) -> float:
    user_idx = model.user_to_idx.get(int(user_id))
    item_idx = model.item_to_idx.get(int(item_id))
    if user_idx is None or item_idx is None:
        return model.global_mean
    return float(
        model.global_mean
        + model.user_bias[user_idx]
        + model.item_bias[item_idx]
        + model.user_factors[user_idx] @ model.item_factors[item_idx]
    )


def generate_mf_candidates(
    model: MFModel,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    candidates_per_user: int = 100,
    max_users: int | None = None,
) -> pd.DataFrame:
    movie_lookup = movies.set_index("movieId").to_dict("index")
    all_model_items = set(model.item_to_idx.keys())
    train_items_by_user = train_ratings.groupby("userId")["movieId"].apply(set).to_dict()
    test_by_user = test_ratings.set_index("userId")["movieId"].to_dict()
    users = [user for user in test_ratings["userId"].tolist() if int(user) in model.user_to_idx]
    if max_users is not None:
        users = users[:max_users]

    rows = []
    for user_id in users:
        seen = train_items_by_user.get(user_id, set())
        candidate_items = list(all_model_items - seen)
        scores = [
            (item_id, predict_rating(model, int(user_id), int(item_id)))
            for item_id in candidate_items
        ]
        scores.sort(key=lambda pair: pair[1], reverse=True)
        top_scores = scores[:candidates_per_user]
        heldout_item = int(test_by_user[user_id])

        if heldout_item not in {item_id for item_id, _ in top_scores} and heldout_item in all_model_items:
            top_scores.append((heldout_item, predict_rating(model, int(user_id), heldout_item)))

        for item_id, predicted_rating in top_scores:
            metadata = movie_lookup.get(int(item_id), {})
            rows.append(
                {
                    "user_id": int(user_id),
                    "item_id": str(item_id),
                    "movieId": int(item_id),
                    "title": str(metadata.get("title", f"Movie {item_id}")),
                    "genres": str(metadata.get("genres", "")),
                    "overview": str(metadata.get("overview", metadata.get("title", ""))),
                    "baseline_score": normalize_score((predicted_rating - 0.5) / 4.5),
                    "predicted_rating": predicted_rating,
                    "heldout_item": heldout_item,
                    "is_relevant": int(item_id == heldout_item),
                }
            )

    return pd.DataFrame(rows)


def build_user_profile(
    user_id: int,
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 4,
) -> UserProfile:
    user_items = train_ratings[train_ratings["userId"] == user_id]
    merged = user_items.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    genre_scores: dict[str, float] = {}
    for row in merged.itertuples(index=False):
        for genre in str(row.genres).split("|"):
            genre = genre.strip()
            if not genre or genre == "(no genres listed)":
                continue
            genre_scores[genre] = genre_scores.get(genre, 0.0) + float(row.rating)
    preferred = sorted(genre_scores, key=genre_scores.get, reverse=True)[:top_n]
    return UserProfile(user_id=str(user_id), preferred_genres=tuple(preferred))


def load_movielens_with_optional_metadata(
    movielens_dir: str,
    metadata_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(f"{movielens_dir}/ratings.csv")
    movies = pd.read_csv(f"{movielens_dir}/movies.csv")
    if metadata_path:
        metadata = pd.read_csv(metadata_path)
        movies = movies.merge(metadata, on="movieId", how="left", suffixes=("", "_tmdb"))
        if "overview" not in movies:
            movies["overview"] = movies["title"]
        else:
            movies["overview"] = movies["overview"].fillna(movies["title"])
    else:
        movies["overview"] = movies["title"] + " " + movies["genres"].str.replace("|", " ", regex=False)
    return ratings, movies
