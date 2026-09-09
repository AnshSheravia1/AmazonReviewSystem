# Amazon Review Intelligence API

A FastAPI service built on the analysis from
`review-sentiment-analysis-and-recommender-system.ipynb`: it scores review
text with VADER sentiment analysis, and recommends books using cosine
similarity over a genre + review-sentiment feature space. The notebook's
data-cleaning, sentiment, and recommender logic has been extracted into
`app/data_pipeline.py`, `app/sentiment.py`, and `app/recommender.py`, and is
precomputed once into artifacts the API loads at startup instead of
recomputing on every boot. A small static frontend (`frontend/`) is served
by the same app for trying it out in a browser without touching `curl`.

## Project layout

```
app/
  main.py           FastAPI app + routes, serves frontend/ as static files
  data_pipeline.py  loading, cleaning, feature engineering (the notebook's logic)
  sentiment.py      VADER wrapper
  recommender.py    cosine-similarity recommender
  models.py         Pydantic request/response models
frontend/
  index.html, style.css, app.js   plain HTML/CSS/JS demo UI, no build step
scripts/
  download_data.py         pulls the raw Kaggle CSVs into data/raw/
  generate_sample_data.py  schema-identical synthetic data for local dev, written to data/sample_raw/
  build_artifacts.py       runs the real pipeline, writes data/processed/*
data/
  raw/         raw Kaggle CSVs (gitignored, not committed — ~2.8 GB)
  sample_raw/  small synthetic CSVs for local dev (gitignored)
  processed/   books_catalog.parquet + feature_matrix.npz the API loads at startup
tests/
  test_api.py
```

## Dataset

[Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)
on Kaggle: `books_data.csv` (~212K books, metadata) joined with
`Books_rating.csv` (~3M reviews, ~2.8 GB) on `Title`. The full raw CSVs are
not committed to this repo — regenerate them locally with the steps below.
The committed `data/processed/` artifacts (~1.7 MB total) were built from
the **real** dataset: 25,380 unique books, sampled down from the full 3M
reviews the same way the notebook does (`sample(n=50000, random_state=42)`).

## Regenerating the artifacts from real data

```bash
pip install -r requirements-dev.txt   # adds kagglehub, needed only for download_data.py

# 1. Download the raw CSVs into data/raw/ (public dataset — no Kaggle
#    credentials were needed when this was last run, but if kagglehub asks,
#    set up ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY)
python scripts/download_data.py

# 2. Run the real pipeline and write data/processed/books_catalog.parquet
#    and data/processed/feature_matrix.npz
python scripts/build_artifacts.py --sample-size 50000
```

`--sample-size` mirrors the notebook's `subset = merged.sample(n=50000, ...)`
step — pass `--sample-size 0` to process every row instead (slow: VADER runs
once per review across the full 3M rows, and produces a much larger catalog).

To instead regenerate the small local sample used for fast local dev/tests:

```bash
python scripts/generate_sample_data.py
python scripts/build_artifacts.py \
  --books-data data/sample_raw/books_data.csv \
  --ratings data/sample_raw/Books_rating.csv \
  --sample-size 0
```

## Running the API locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/` for the demo frontend, or
`http://127.0.0.1:8000/docs` for interactive API docs. (`GET /api` returns
the old plain-JSON root message, now that `/` serves the frontend.)

## Frontend

`frontend/` is a plain HTML/CSS/JS page (no build step, no framework) served
by the FastAPI app itself via `StaticFiles`, mounted at `/`. It has a
textbox for sentiment analysis and a title-search box (backed by the new
`GET /books?q=` endpoint) for picking a book and viewing its recommendations.
It's just a thin client over the existing JSON endpoints — open the browser
devtools network tab to see it calling `/sentiment`, `/books`, and
`/recommend/{book_id}` directly. CORS is wide open (`allow_origins=["*"]`)
so it also works if served from anywhere else during development.

## Example requests

Captured from a real run against the full processed catalog (25,380 books):

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","books_loaded":25380}

curl -s -X POST http://127.0.0.1:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this book, it was amazing and wonderful!"}'
# {"text":"I absolutely loved this book, it was amazing and wonderful!","compound_score":0.9183,"sentiment":"positive"}

curl -s -X POST http://127.0.0.1:8000/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This was a terrible, boring, awful book."}'
# {"text":"This was a terrible, boring, awful book.","compound_score":-0.8126,"sentiment":"negative"}

curl -s -X POST http://127.0.0.1:8000/sentiment -d '{"text": "   "}'
# {"detail":"Text cannot be empty."}

curl -s "http://127.0.0.1:8000/books?q=dune&limit=3"
# [
#   {"book_id":"0553526677","title":"House Corrino (Dune: House Trilogy, Book 3)","genre":"Fiction"},
#   {"book_id":"0765305852","title":"The Butlerian Jihad (Legends of Dune, Book 1)","genre":"Fiction"},
#   {"book_id":"0765305860","title":"The Machine Crusade (Legends of Dune, Book 2)","genre":"Fiction"}
# ]

curl -s "http://127.0.0.1:8000/recommend/1573451657?top_n=3"
# [
#   {"book_id":"0310806291","title":"Footprints","genre":"Biography & Autobiography","similarity_score":1.0},
#   {"book_id":"0307338282","title":"For Laci: A Mother's Story of Love, Loss, and Justice","genre":"Biography & Autobiography","similarity_score":1.0},
#   {"book_id":"0310208068","title":"Spiritual Lives of the Great Composers","genre":"Biography & Autobiography","similarity_score":1.0}
# ]

curl -s "http://127.0.0.1:8000/recommend/does-not-exist"
# {"detail":"book_id does-not-exist not found."}
```

(`book_id` is the Amazon `Id` field from the source data — an ASIN-like
string, not a small integer.)

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Covers: `/sentiment` on clearly positive/negative/empty input, and
`/recommend/{book_id}` on a valid id (returns `top_n` results, none equal to
the input book) and an invalid id (404). All 6 tests pass against the real
25,380-book catalog.

## Notes on the recommender

The notebook builds a `Title x Genre` pivot table of rating-quality scores
and runs cosine similarity over it (`recommend_books_by_genre`, cell 36).
This service adapts that in two ways:

- It pivots on `book_id` instead of `Title`, since two different books can
  share a title but the API needs a stable per-book key for
  `/recommend/{book_id}`.
- At real-dataset scale (25,380 unique books) a precomputed dense N x N
  similarity matrix is ~4.8 GB and doesn't fit in memory on a typical
  machine. Because each book's feature vector is one-hot-by-genre (a single
  nonzero entry, at its own genre column, holding its rating-quality
  ordinal), the *feature matrix* itself is tiny and sparse — that's what
  gets precomputed and persisted (`feature_matrix.npz`). Cosine similarity
  for one queried book against the whole catalog is then a single O(n)
  sparse matrix-vector product computed per-request
  (`recommender.recommend_similar`), rather than an O(n²) matrix computed
  once. A side effect worth knowing: since the feature vectors are one-hot,
  cosine similarity between any two books in the *same* genre is always
  exactly 1.0 — this recommender is effectively genre-filtering, matching
  what the notebook's own `recommend_books_by_genre` does.

## Environment note

This was built and validated in a sandbox with no interactive Kaggle login;
`kagglehub.dataset_download` still succeeded because this dataset is public.
If you hit an auth prompt in a different environment, set up Kaggle API
credentials per https://github.com/Kaggle/kaggle-api#api-credentials.
