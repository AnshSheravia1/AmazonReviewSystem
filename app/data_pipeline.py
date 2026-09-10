"""Data loading, cleaning, and feature engineering.

Reproduces the pipeline built in
review-sentiment-analysis-and-recommender-system.ipynb (cells 2, 9, 13-27,
31-33) against the real Kaggle "Amazon Books Reviews" CSVs:
  - book_data:    Title, description, authors, image, previewLink,
                   publisher, publishedDate, infoLink, categories
  - book_ratings: Id, Title, Price, User_id, profileName,
                   review/helpfulness, review/score, review/time,
                   review/summary, review/text

Two granularities are produced:
  - build_review_level_frame: one row per review, with per-review VADER
    sentiment, cleaned genre/author, kept around for genre-level keyword
    extraction (app/keywords.py), which needs individual review text.
  - build_book_catalog: the review-level frame collapsed to one row per
    book, which is what the recommender and API serve.
"""
from __future__ import annotations

import pandas as pd

from .sentiment import label_from_compound, score_text, sentiment_category

FILL_UNKNOWN_COLS = [
    "description", "authors", "image", "previewLink",
    "publisher", "publishedDate", "infoLink", "categories",
]

DROP_COLS = ["review/time", "Price", "image", "previewLink", "infoLink"]

RATING_ORDINAL_MAP = {"Poor": 1, "Average": 2, "Good": 3, "Very Good": 4, "Must Read": 5}


def load_and_merge(books_data_path: str, ratings_path: str) -> pd.DataFrame:
    # Only the columns build_review_level_frame actually uses are loaded —
    # the full 3M-row ratings CSV (with review/text, review/helpfulness,
    # etc. included) is too large to hold in memory in one pass otherwise.
    book_data = pd.read_csv(books_data_path, usecols=["Title", "authors", "categories"])
    book_ratings = pd.read_csv(
        ratings_path,
        usecols=["Id", "Title", "review/score", "review/summary"],
        dtype={"Id": "string", "review/score": "float32"},
    )

    for col in FILL_UNKNOWN_COLS:
        if col in book_data.columns:
            book_data[col] = book_data[col].fillna("Unknown")

    merged = pd.merge(book_data, book_ratings, on="Title")
    merged = merged.drop(columns=[c for c in DROP_COLS if c in merged.columns])
    return merged


def _clean_bracketed(series: pd.Series) -> pd.Series:
    # categories/authors come in as "['Some Genre']" string-encoded lists
    return series.astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()


def _categorize_rating(avg_rating: float) -> str:
    # Notebook's original bucketing (strict < / > on both sides) left
    # avg_rating in {2, 3, 4} unmapped; inclusive lower bounds fix that so
    # every rating in (0, 5] lands in a bucket.
    if avg_rating <= 1:
        return "Poor"
    elif avg_rating <= 2:
        return "Average"
    elif avg_rating <= 3:
        return "Good"
    elif avg_rating <= 4:
        return "Very Good"
    else:
        return "Must Read"


def build_review_level_frame(
    merged: pd.DataFrame,
    sample_size: int | None = 50000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cleans the merged review-level frame and attaches per-review VADER
    sentiment, cleaned Genre/Author, and a positive/negative/neutral label —
    the granularity app/keywords.py needs (per-review text, not per-book)."""
    df = merged.copy()

    df["ratingsCount"] = df.groupby("Id")["Id"].transform("count")
    df["avgRating"] = df.groupby("Id")["review/score"].transform("mean")

    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    df["review/summary"] = df["review/summary"].fillna("Unknown")
    df.loc[df["review/summary"].astype(str).str.strip() == "", "review/summary"] = "Unknown"
    df["review_compound"] = df["review/summary"].apply(score_text)
    df["sentiment_label"] = df["review_compound"].apply(label_from_compound)

    df["Genre"] = _clean_bracketed(df["categories"])
    df["Author"] = _clean_bracketed(df["authors"])

    # Backfill unknown genres from the author's most common known genre.
    known = df[df["Genre"] != "Unknown"]
    author_genre = known.groupby("Author")["Genre"].agg(
        lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA
    )
    author_genre_map = author_genre.dropna().to_dict()
    unknown_mask = df["Genre"] == "Unknown"
    df.loc[unknown_mask, "Genre"] = (
        df.loc[unknown_mask, "Author"].map(author_genre_map).fillna("Unknown")
    )

    return df.reset_index(drop=True)


def build_book_catalog(review_df: pd.DataFrame) -> pd.DataFrame:
    """Collapses the review-level frame into one row per book_id (Amazon's
    Id), with an aggregate sentiment score, a pooled review-text blob (for
    the content-based similarity signal), and the genre/rating features
    used by the recommender."""
    catalog = (
        review_df.groupby("Id")
        .agg(
            title=("Title", "first"),
            genre=("Genre", "first"),
            author=("Author", "first"),
            avg_rating=("avgRating", "first"),
            avg_sentiment_score=("review_compound", "mean"),
            review_count=("review_compound", "size"),
            review_text_blob=("review/summary", lambda s: " ".join(s.astype(str))),
        )
        .reset_index()
        .rename(columns={"Id": "book_id"})
    )

    catalog["sentiment_category"] = catalog["avg_sentiment_score"].apply(sentiment_category)
    catalog["rating_category"] = catalog["avg_rating"].apply(_categorize_rating)
    catalog["rating_ordinal"] = catalog["rating_category"].map(RATING_ORDINAL_MAP)
    catalog["genre_encoded"] = catalog["genre"].astype("category").cat.codes

    return catalog.reset_index(drop=True)
