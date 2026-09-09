"""Downloads the raw Kaggle "Amazon Books Reviews" dataset used by the
notebook (books_data.csv + Books_rating.csv) into data/raw/.

Requires Kaggle API credentials: either ~/.kaggle/kaggle.json, or the
KAGGLE_USERNAME / KAGGLE_KEY environment variables. See
https://github.com/Kaggle/kaggle-api#api-credentials

Requires the `kagglehub` package (pip install kagglehub); not part of the
service's own runtime requirements, so it's kept out of requirements.txt.
"""
import shutil
from pathlib import Path

import kagglehub

DATASET = "mohamedbakhet/amazon-books-reviews"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download_dataset(dest_dir: str = str(RAW_DIR)) -> Path:
    """
    Downloads the Amazon Books Reviews dataset via kagglehub and copies
    the CSVs into the project's data/raw directory (kagglehub caches
    downloads outside the repo by default, in ~/.cache/kagglehub).
    """
    path = kagglehub.dataset_download(DATASET)
    print(f"kagglehub cached dataset at: {path}")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    src_path = Path(path)
    for csv_file in src_path.glob("*.csv"):
        shutil.copy(csv_file, dest / csv_file.name)
        print(f"Copied {csv_file.name} -> {dest}")

    return dest


if __name__ == "__main__":
    download_dataset()
