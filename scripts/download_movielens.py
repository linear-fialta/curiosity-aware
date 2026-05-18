from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATASET_NAME = "ml-latest-small"
DATASET_URL = f"https://files.grouplens.org/datasets/movielens/{DATASET_NAME}.zip"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / f"{DATASET_NAME}.zip"
    extract_path = RAW_DIR / DATASET_NAME

    if not zip_path.exists():
        print(f"Downloading {DATASET_URL}")
        with urllib.request.urlopen(DATASET_URL) as response:
            with zip_path.open("wb") as output:
                shutil.copyfileobj(response, output)
    else:
        print(f"Using existing archive: {zip_path}")

    if not extract_path.exists():
        print(f"Extracting to {RAW_DIR}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(RAW_DIR)
    else:
        print(f"Using existing extracted data: {extract_path}")

    expected_files = ["ratings.csv", "movies.csv", "links.csv"]
    for file_name in expected_files:
        path = extract_path / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing expected MovieLens file: {path}")

    print(f"MovieLens data ready at {extract_path}")


if __name__ == "__main__":
    main()

