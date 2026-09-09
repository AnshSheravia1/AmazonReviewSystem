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
