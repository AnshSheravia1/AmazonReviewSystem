"""Cosine-similarity recommender.

Reproduces the notebook's recommender (cells 31-35): a genre x rating-quality
feature per book, compared with cosine similarity.

At real-dataset scale the catalog has ~25K unique books, so a precomputed
dense N x N similarity matrix (~4.8 GB of float64) doesn't fit in memory.
The feature vectors are one-hot-by-genre (each book has a single nonzero
entry, at its own genre column, holding its rating-quality ordinal), so the
feature matrix itself is tiny and sparse — it's what gets precomputed and
persisted. Cosine similarity for one query book against the whole catalog is
then a single O(n) sparse matrix-vector product, computed on demand in
recommend_similar() rather than precomputed for every pair up front.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


def build_feature_matrix(catalog: pd.DataFrame) -> sparse.csr_matrix:
    n = len(catalog)
    k = int(catalog["genre_encoded"].max()) + 1 if n else 0
    rows = np.arange(n)
    cols = catalog["genre_encoded"].to_numpy()
    data = catalog["rating_ordinal"].to_numpy(dtype="float64")
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, k))


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
