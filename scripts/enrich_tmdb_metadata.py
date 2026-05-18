from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MOVIELENS_DIR = ROOT / "data" / "raw" / "ml-latest-small"
EXTERNAL_DIR = ROOT / "data" / "external"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
CACHE_PATH = EXTERNAL_DIR / "tmdb_movie_cache.csv"
OUTPUT_PATH = EXTERNAL_DIR / "tmdb_movies.csv"


def fetch_tmdb_movie(tmdb_id: str, api_key: str, retries: int = 4) -> dict:
    query = urllib.parse.urlencode({"api_key": api_key, "language": "en-US"})
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as exc:
            wait_seconds = min(2**attempt, 30)
            print(
                f"Retrying TMDb id {tmdb_id} after network error "
                f"({attempt + 1}/{retries}): {exc}"
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to fetch TMDb id {tmdb_id} after {retries} retries.")


def main() -> None:
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("Set TMDB_API_KEY before running this script.")

    links_path = MOVIELENS_DIR / "links.csv"
    movies_path = MOVIELENS_DIR / "movies.csv"
    if not links_path.exists() or not movies_path.exists():
        raise FileNotFoundError("Run scripts/download_movielens.py before TMDb enrichment.")

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    links = pd.read_csv(links_path).dropna(subset=["tmdbId"])
    movies = pd.read_csv(movies_path)
    links["tmdbId"] = links["tmdbId"].astype(int).astype(str)

    rows = _load_cached_rows()
    fetched_movie_ids = {int(row["movieId"]) for row in rows}
    for idx, row in links.iterrows():
        movie_id = int(row["movieId"])
        if movie_id in fetched_movie_ids:
            continue

        tmdb_id = row["tmdbId"]
        try:
            payload = fetch_tmdb_movie(tmdb_id, api_key)
        except urllib.error.HTTPError as exc:
            print(f"Skipping TMDb id {tmdb_id}: HTTP {exc.code}")
            continue
        except RuntimeError as exc:
            print(f"Stopping early: {exc}")
            break

        poster_path = payload.get("poster_path")
        rows.append(
            {
                "movieId": movie_id,
                "tmdbId": tmdb_id,
                "overview": payload.get("overview", ""),
                "tagline": payload.get("tagline", ""),
                "release_date": payload.get("release_date", ""),
                "popularity": payload.get("popularity"),
                "poster_url": f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else "",
            }
        )
        fetched_movie_ids.add(movie_id)

        if len(rows) % 25 == 0:
            _write_cache(rows)
            print(f"Cached {len(rows)} TMDb records")
        time.sleep(0.15)

    _write_cache(rows)
    enriched = movies.merge(pd.DataFrame(rows), on="movieId", how="inner")
    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


def _load_cached_rows() -> list[dict]:
    if not CACHE_PATH.exists():
        return []
    cached = pd.read_csv(CACHE_PATH)
    print(f"Resuming from cache with {len(cached)} TMDb records")
    return cached.to_dict("records")


def _write_cache(rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).drop_duplicates(subset=["movieId"]).to_csv(CACHE_PATH, index=False)


if __name__ == "__main__":
    main()
