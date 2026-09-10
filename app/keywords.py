"""Genre-level keyword extraction: "what are people actually praising or
complaining about" in each genre, mined from review text rather than
declared upfront.

Approach: pool every positive review's summary text within a genre into one
"document", and every negative review's summary text into another; treat
each (genre, sentiment) pool as one document in a corpus of every such pool.
TF-IDF's IDF term then does the real work — words common everywhere
("book", "read", "good") get downweighted since they appear in nearly every
pool, while words distinctive to e.g. "Cooking / positive" or
"Mystery / negative" rise to the top. This is a standard, well-understood
technique (TF-IDF over pooled per-group documents) and needs no topic-count
tuning the way LDA would, which is why it was chosen over LDA here.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_REVIEWS_PER_BUCKET = 20
TOP_K = 12
MAX_FEATURES = 8000


def extract_genre_keywords(
    review_df: pd.DataFrame,
    min_reviews: int = MIN_REVIEWS_PER_BUCKET,
    top_k: int = TOP_K,
) -> dict[str, dict[str, list[str]]]:
    """Returns {genre: {"positive": [...top terms...], "negative": [...]}}
    for every genre with at least `min_reviews` reviews of each sentiment.
    "Unknown" genre and "neutral" sentiment are excluded — the former isn't
    a real genre, the latter isn't praise or complaint."""
    df = review_df[
        (review_df["Genre"] != "Unknown") & (review_df["sentiment_label"] != "neutral")
    ]

    bucket_counts = df.groupby(["Genre", "sentiment_label"]).size()
    valid_buckets = bucket_counts[bucket_counts >= min_reviews].index

    if len(valid_buckets) == 0:
        return {}

    pooled = (
        df.groupby(["Genre", "sentiment_label"])["review/summary"]
        .agg(lambda s: " ".join(s.astype(str)))
        .reset_index()
    )
    pooled = pooled.set_index(["Genre", "sentiment_label"]).loc[list(valid_buckets)].reset_index()

    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=1
    )
    tfidf = vectorizer.fit_transform(pooled["review/summary"])
    terms = vectorizer.get_feature_names_out()

    results: dict[str, dict[str, list[str]]] = {}
    for row_idx, row in pooled.iterrows():
        row_vec = tfidf[row_idx].toarray().ravel()
        top_idx = row_vec.argsort()[::-1][:top_k]
        keywords = [terms[i] for i in top_idx if row_vec[i] > 0]
        results.setdefault(row["Genre"], {})[row["sentiment_label"]] = keywords

    return results
