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


def fetch_tmdb_movie(tmdb_id: str, api_key: str) -> dict:
    query = urllib.parse.urlencode({"api_key": api_key, "language": "en-US"})
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key:
        raise RuntimeError("Set TMDB_API_KEY before running this script.")

    links_path = MOVIELENS_DIR / "links.csv"
    movies_path = MOVIELENS_DIR / "movies.csv"
    if not links_path.exists() or not movies_path.exists():
        raise FileNotFoundError("Run scripts/download_movielens.py before TMDb enrichment.")

    links = pd.read_csv(links_path).dropna(subset=["tmdbId"])
    movies = pd.read_csv(movies_path)
    links["tmdbId"] = links["tmdbId"].astype(int).astype(str)

    rows = []
    for idx, row in links.iterrows():
        movie_id = int(row["movieId"])
        tmdb_id = row["tmdbId"]
        try:
            payload = fetch_tmdb_movie(tmdb_id, api_key)
        except urllib.error.HTTPError as exc:
            print(f"Skipping TMDb id {tmdb_id}: HTTP {exc.code}")
            continue

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

        if (idx + 1) % 100 == 0:
            print(f"Fetched {idx + 1} TMDb records")
        time.sleep(0.05)

    enriched = movies.merge(pd.DataFrame(rows), on="movieId", how="inner")
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXTERNAL_DIR / "tmdb_movies.csv"
    enriched.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

