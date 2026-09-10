"""Cosine-similarity recommender.

Builds on the notebook's recommender (cells 31-35: a genre x rating-quality
feature per book, compared with cosine similarity), with one deliberate fix.

The notebook's feature vector is one-hot-by-genre (each book has a single
nonzero entry, at its own genre column, holding its rating-quality ordinal).
Cosine similarity between two such vectors is *exactly* 1.0 whenever the
books share a genre and 0.0 otherwise — the recommender is really just a
genre filter, with no way to tell two same-genre books apart. To fix that
without abandoning the notebook's approach, a genuine content signal is
concatenated on: TF-IDF over each book's pooled review-summary text. Two
books in the same genre now score anywhere in (0, 1) depending on how
similar their reviews actually read, and the genre/rating blocks still
pull same-genre, similarly-rated books closer than cross-genre ones.

At real-dataset scale the catalog has ~25K unique books, so a precomputed
dense N x N similarity matrix (~4.8 GB of float64) doesn't fit in memory.
The combined feature matrix stays sparse and is what's precomputed and
persisted; cosine similarity for one queried book against the whole catalog
is a single O(n) sparse matrix-vector product, computed on demand in
recommend_similar() rather than precomputed for every pair up front.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Relative weight of each feature block once every block is independently
# L2-normalized. Genre and rating are coarse, near-binary signals; text
# similarity is the one continuous, discriminative signal, so it's weighted
# highest — high enough that same-genre books are no longer tied at 1.0, but
# not so high that genre stops mattering at all.
GENRE_WEIGHT = 1.0
RATING_WEIGHT = 1.0
TEXT_WEIGHT = 2.0

TFIDF_MAX_FEATURES = 20000


def _genre_block(catalog: pd.DataFrame) -> sparse.csr_matrix:
    n = len(catalog)
    k = int(catalog["genre_encoded"].max()) + 1 if n else 0
    rows = np.arange(n)
    cols = catalog["genre_encoded"].to_numpy()
    data = np.ones(n, dtype="float64")
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, k))


def _rating_block(catalog: pd.DataFrame) -> sparse.csr_matrix:
    # Single column: rating_ordinal scaled to [0, 1]. Kept as its own block
    # (rather than folded into the genre one-hot as the notebook does) so
    # its weight can be tuned independently.
    values = (catalog["rating_ordinal"].to_numpy(dtype="float64") / 5.0).reshape(-1, 1)
    return sparse.csr_matrix(values)


def build_content_features(
    catalog: pd.DataFrame, max_features: int = TFIDF_MAX_FEATURES
) -> sparse.csr_matrix:
    """TF-IDF over each book's pooled review-summary text
    (catalog['review_text_blob']) — the genuine content-based signal.

    max_features (when the vocabulary exceeds it) keeps the *most frequent*
    remaining terms, so it must stay a high safety-net cap, not a tight one:
    a tight cap keeps generic high-frequency words and drops specific,
    discriminative ones (exactly backwards for a similarity signal). min_df=2
    drops words that appear in only one book's pooled text, since a term no
    other book shares can never contribute to a similarity score anyway.
    max_df=0.5 drops words so common (across more than half the catalog)
    that they can't discriminate between books either.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=max_features, min_df=2, max_df=0.5
    )
    return vectorizer.fit_transform(catalog["review_text_blob"].fillna(""))


def build_feature_matrix(catalog: pd.DataFrame) -> sparse.csr_matrix:
    # NOTE: the rating block is a single column, so L2-normalizing it row-by-row
    # (like the genre and text blocks) would divide every value by itself and
    # collapse it to a constant 1.0 for every book — silently erasing the signal.
    # It's already scaled to [0, 1] by _rating_block, so it's used as-is.
    genre = normalize(_genre_block(catalog)) * GENRE_WEIGHT
    rating = _rating_block(catalog) * RATING_WEIGHT
    text = normalize(build_content_features(catalog)) * TEXT_WEIGHT
    return sparse.hstack([genre, rating, text], format="csr")


def recommend_similar(
    feature_matrix: sparse.csr_matrix, idx: int, top_n: int = 3
) -> list[tuple[int, float]]:
    """Returns up to top_n (row_index, similarity_score) pairs for the books
    most similar to catalog row `idx`, excluding `idx` itself."""
    query = feature_matrix[idx]
    sims = cosine_similarity(query, feature_matrix)[0]
    order = np.argsort(-sims)

    results = []
    for i in order:
        if i == idx:
            continue
        results.append((int(i), float(sims[i])))
        if len(results) >= top_n:
            break
    return results
