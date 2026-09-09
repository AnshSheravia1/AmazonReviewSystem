from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from scipy import sparse

from .models import (
    HealthResponse,
    RecommendationResponse,
    SentimentRequest,
    SentimentResponse,
)
from .recommender import recommend_similar
from .sentiment import analyze

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CATALOG_PATH = DATA_DIR / "books_catalog.parquet"
FEATURES_PATH = DATA_DIR / "feature_matrix.npz"

state: dict = {"catalog": None, "features": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["catalog"] = pd.read_parquet(CATALOG_PATH)
    state["features"] = sparse.load_npz(FEATURES_PATH)
    yield
    state["catalog"] = None
    state["features"] = None


app = FastAPI(
    title="Amazon Review Intelligence API",
    description="Sentiment analysis and genre-based book recommendations, mined from Amazon review data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Amazon Review Intelligence API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse)
def health():
    catalog = state["catalog"]
    return HealthResponse(
        status="ok" if catalog is not None else "not_loaded",
        books_loaded=int(len(catalog)) if catalog is not None else 0,
    )


@app.post("/sentiment", response_model=SentimentResponse)
def get_sentiment(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    compound, label = analyze(request.text)
    return SentimentResponse(text=request.text, compound_score=compound, sentiment=label)


@app.get("/recommend/{book_id}", response_model=list[RecommendationResponse])
def recommend(book_id: str, top_n: int = 3):
    catalog = state["catalog"]
    features = state["features"]

    matches = catalog.index[catalog["book_id"] == book_id]
    if len(matches) == 0:
        raise HTTPException(status_code=404, detail=f"book_id {book_id} not found.")

    idx = matches[0]
    top_matches = recommend_similar(features, idx, top_n=top_n)

    results = []
    for match_idx, score in top_matches:
        row = catalog.iloc[match_idx]
        results.append(
            RecommendationResponse(
                book_id=str(row["book_id"]),
                title=row["title"],
                genre=row["genre"],
                similarity_score=round(score, 4),
            )
        )
    return results
