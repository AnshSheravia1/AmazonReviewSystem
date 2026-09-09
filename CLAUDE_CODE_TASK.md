# Task: Wire up FastAPI service to real AmazonReviewSystem notebook logic

## Context

This repo (`AmazonReviewSystem`) currently contains a single Jupyter notebook
(`review-sentiment-analysis-and-recommender-system.ipynb`) that:
- Runs VADER sentiment analysis on a 3-million-row Amazon reviews dataset
- Builds a genre-based book recommender using cosine similarity over
  combined genre + sentiment features
- Uses Pandas/NumPy for data manipulation, Matplotlib/Seaborn for visualization

I've been given a scaffold FastAPI service (`main.py` + `requirements.txt`,
included below) that exposes this logic as two working REST endpoints:
`POST /sentiment` and `GET /recommend/{book_id}`. It currently runs on a
small hardcoded placeholder DataFrame so it's runnable out of the box.

## Goal

Replace the placeholder data and wire the API up to the **real** notebook
logic, so the deployed service reflects the actual trained
sentiment/recommender pipeline, not fake data.

## Steps

1. **Read the existing notebook** (`review-sentiment-analysis-and-recommender-system.ipynb`)
   in this repo. Identify:
   - The exact data loading step (file path / Kaggle dataset reference)
   - The cleaning/preprocessing steps applied before sentiment analysis
   - The exact column names used for genre, sentiment score, book/product
     identifier, and title
   - How the cosine similarity feature matrix is actually constructed
     (confirm it matches what's in `main.py`'s `build_feature_matrix`,
     or adjust `main.py` to match the real approach)

2. **Extract the notebook logic into reusable Python modules**, rather than
   leaving everything in `main.py`. Suggested structure:
   ```
   /app
     main.py              # FastAPI app + route definitions only
     data_pipeline.py      # data loading, cleaning, feature engineering
     sentiment.py           # VADER wrapper logic
     recommender.py         # cosine similarity recommender logic
     models.py               # Pydantic request/response models
   /data
     (processed dataset, or a script to regenerate it — do NOT commit
     the raw 3M-row CSV if it's large; add a data-download or
     data-processing script instead, and .gitignore the raw file)
   requirements.txt
   README.md
   ```

3. **Precompute what should be precomputed.** Running VADER sentiment
   analysis and building the full similarity matrix on every server startup
   across 3M rows will be slow. Precompute the processed DataFrame
   (with sentiment scores already attached) and the similarity matrix once,
   save them (e.g. `.parquet` for the DataFrame, `.npy` for the matrix),
   and have the FastAPI app load these precomputed artifacts on startup
   instead of recomputing from raw data every time.

4. **Update `main.py`** to:
   - Load the precomputed data/artifacts on startup (not the placeholder
     `books_df`)
   - Keep the same two endpoints (`/sentiment`, `/recommend/{book_id}`),
     same request/response shape, unless the real column names require
     small adjustments — if so, update the Pydantic models to match
   - Add a `/health` endpoint that returns basic status + how many
     books/items are loaded, useful for confirming the real data loaded
     correctly

5. **Add basic tests** (`pytest` + `fastapi.testclient`) covering:
   - `/sentiment` with a clearly positive review, clearly negative review,
     and empty string (should 400)
   - `/recommend/{book_id}` with a valid ID (should return `top_n` results,
     none of which is the input book itself) and an invalid ID (should 404)

6. **Update the README** to include:
   - What the service does, in 2-3 sentences
   - How to regenerate the precomputed artifacts from raw data
   - How to run the API locally (`uvicorn main:app --reload`)
   - Example `curl` requests for both endpoints, with real example output
   - A note on the dataset source (Kaggle, size, link)
   - 1-2 screenshots or pasted output showing example sentiment scores
     and example recommendations, so someone skimming the README sees
     real results without needing to run anything

7. **Do NOT commit large data files or API keys.** Make sure `.gitignore`
   covers the raw dataset, any `.env` file, and precomputed artifacts if
   they're large (>50MB) — instead provide the script to regenerate them.

## Current scaffold files (starting point)

### main.py
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Amazon Review Intelligence API",
    description="Sentiment analysis and genre-based book recommendations, mined from Amazon review data.",
    version="1.0.0",
)

analyzer = SentimentIntensityAnalyzer()

# TODO: replace this with real data loading — see task instructions above.
books_df = pd.DataFrame({
    "book_id": [1, 2, 3, 4, 5],
    "title": ["Dune", "Foundation", "Neuromancer", "The Hobbit", "1984"],
    "genre": ["sci-fi", "sci-fi", "sci-fi", "fantasy", "dystopian"],
    "avg_sentiment_score": [0.85, 0.78, 0.65, 0.90, 0.72],
})


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    genre_dummies = pd.get_dummies(df["genre"])
    features = pd.concat([genre_dummies, df["avg_sentiment_score"]], axis=1)
    return features.values


feature_matrix = build_feature_matrix(books_df)
similarity_matrix = cosine_similarity(feature_matrix)


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    text: str
    compound_score: float
    sentiment: str


class RecommendationResponse(BaseModel):
    book_id: int
    title: str
    genre: str
    similarity_score: float


@app.get("/")
def root():
    return {"message": "Amazon Review Intelligence API. See /docs for usage."}


@app.post("/sentiment", response_model=SentimentResponse)
def get_sentiment(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    scores = analyzer.polarity_scores(request.text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return SentimentResponse(text=request.text, compound_score=compound, sentiment=sentiment)


@app.get("/recommend/{book_id}", response_model=list[RecommendationResponse])
def recommend(book_id: int, top_n: int = 3):
    if book_id not in books_df["book_id"].values:
        raise HTTPException(status_code=404, detail=f"book_id {book_id} not found.")

    idx = books_df.index[books_df["book_id"] == book_id][0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = [s for s in sim_scores if s[0] != idx]
    sim_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = sim_scores[:top_n]

    results = []
    for match_idx, score in top_matches:
        row = books_df.iloc[match_idx]
        results.append(RecommendationResponse(
            book_id=int(row["book_id"]),
            title=row["title"],
            genre=row["genre"],
            similarity_score=round(float(score), 4),
        ))

    return results
```

### requirements.txt
```
fastapi
uvicorn[standard]
vaderSentiment
pandas
numpy
scikit-learn
pydantic
```

## Acceptance criteria

- [ ] `uvicorn main:app --reload` starts cleanly with real data loaded
      (not the placeholder)
- [ ] `/health` reports the real number of items loaded
- [ ] `/sentiment` and `/recommend/{book_id}` both work against real data,
      verified via `/docs` or `curl`
- [ ] Tests pass (`pytest`)
- [ ] README updated with real example output
- [ ] No large data files or secrets committed to git
