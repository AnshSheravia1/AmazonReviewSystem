"""Regenerates the precomputed catalog, feature matrix, and genre keyword
data that the API loads at startup, by running the real pipeline
(app/data_pipeline.py, app/recommender.py, app/keywords.py) over the raw
CSVs.

Usage:
    python scripts/build_artifacts.py \\
        --books-data data/raw/books_data.csv \\
        --ratings data/raw/Books_rating.csv \\
        --sample-size 50000 \\
        --output-dir data/processed
"""
import argparse
import json
import sys
from pathlib import Path

from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data_pipeline import build_book_catalog, build_review_level_frame, load_and_merge  # noqa: E402
from app.keywords import extract_genre_keywords  # noqa: E402
from app.recommender import build_feature_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books-data", default="data/raw/books_data.csv")
    parser.add_argument("--ratings", default="data/raw/Books_rating.csv")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Row sample size, matching the notebook's subset (0 = use all rows).",
    )
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    sample_size = None if args.sample_size == 0 else args.sample_size

    print(f"Loading and merging {args.books_data} + {args.ratings} ...")
    merged = load_and_merge(args.books_data, args.ratings)

    print(f"Building review-level frame (sample_size={sample_size}) ...")
    reviews = build_review_level_frame(merged, sample_size=sample_size)

    print("Building book catalog ...")
    catalog = build_book_catalog(reviews)

    print("Building feature matrix (genre + rating + TF-IDF review text) ...")
    features = build_feature_matrix(catalog)

    print("Extracting per-genre praise/complaint keywords ...")
    keywords = extract_genre_keywords(reviews)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_dir / "books_catalog.parquet", index=False)
    sparse.save_npz(out_dir / "feature_matrix.npz", features)
    (out_dir / "genre_keywords.json").write_text(json.dumps(keywords, indent=2))

    print(
        f"Saved {len(catalog)} books, a {features.shape} sparse feature matrix, "
        f"and keyword data for {len(keywords)} genres to {out_dir}/"
    )


if __name__ == "__main__":
    main()
