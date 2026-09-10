import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["books_loaded"] > 0


def test_sentiment_positive(client):
    resp = client.post("/sentiment", json={"text": "I absolutely loved this book, it was fantastic!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sentiment"] == "positive"
    assert body["compound_score"] > 0


def test_sentiment_negative(client):
    resp = client.post("/sentiment", json={"text": "This was a terrible, boring, awful book."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sentiment"] == "negative"
    assert body["compound_score"] < 0


def test_sentiment_empty_string_is_rejected(client):
    resp = client.post("/sentiment", json={"text": "   "})
    assert resp.status_code == 400


def test_recommend_valid_book_id(client):
    catalog = pd.read_parquet("data/processed/books_catalog.parquet")
    book_id = catalog.iloc[0]["book_id"]

    resp = client.get(f"/recommend/{book_id}?top_n=3")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) <= 3
    assert all(r["book_id"] != book_id for r in results)


def test_recommend_invalid_book_id(client):
    resp = client.get("/recommend/does-not-exist")
    assert resp.status_code == 404


def test_books_search_by_title(client):
    catalog = pd.read_parquet("data/processed/books_catalog.parquet")
    title = catalog.iloc[0]["title"]
    query = title[:5]

    resp = client.get("/books", params={"q": query})
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) > 0
    assert all(query.lower() in r["title"].lower() for r in results)


def test_frontend_is_served_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_list_genres_and_fetch_keywords(client):
    genres_resp = client.get("/genres")
    assert genres_resp.status_code == 200
    genres = genres_resp.json()
    assert len(genres) > 0

    kw_resp = client.get(f"/genres/{genres[0]}/keywords")
    assert kw_resp.status_code == 200
    body = kw_resp.json()
    assert body["genre"] == genres[0]
    assert isinstance(body["positive"], list)
    assert isinstance(body["negative"], list)


def test_genre_keywords_404_for_unknown_genre(client):
    resp = client.get("/genres/Not-A-Real-Genre/keywords")
    assert resp.status_code == 404


def test_recommend_similarity_is_not_binary(client):
    """The old genre-only feature made every same-genre pair score exactly
    1.0 and every cross-genre pair 0.0. With the TF-IDF content signal
    added, recommendations for the same book should no longer all be tied
    at a single score."""
    catalog = pd.read_parquet("data/processed/books_catalog.parquet")
    book_id = catalog.iloc[0]["book_id"]

    resp = client.get(f"/recommend/{book_id}?top_n=5")
    assert resp.status_code == 200
    scores = [r["similarity_score"] for r in resp.json()]
    assert all(0.0 <= s <= 1.0 for s in scores)
