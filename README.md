# Amazon Review Intelligence API

A FastAPI service built on the analysis from
`review-sentiment-analysis-and-recommender-system.ipynb`: it scores review
text with VADER sentiment analysis, recommends books using cosine similarity
over a genre + rating + review-text feature space, and mines each genre's
reviews with TF-IDF to surface what readers actually praise vs. complain
about. The notebook's data-cleaning and sentiment logic has been extracted
into `app/data_pipeline.py` and `app/sentiment.py`; the recommender
(`app/recommender.py`) and the keyword mining (`app/keywords.py`) go well
beyond what the notebook did — see "Notes on the recommender" and "Genre
Insights" below for why. Everything is precomputed once into artifacts the
API loads at startup instead of recomputing on every boot. A small static
frontend (`frontend/`) is served by the same app for trying it out in a
browser without touching `curl`.

## Project layout

```
app/
  main.py           FastAPI app + routes, serves frontend/ as static files
  data_pipeline.py  loading, cleaning, feature engineering (the notebook's logic)
  sentiment.py      VADER wrapper
  recommender.py    genre + rating + TF-IDF cosine-similarity recommender
  keywords.py       per-genre praise/complaint keyword extraction (TF-IDF)
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
  processed/   books_catalog.parquet, feature_matrix.npz, genre_keywords.json
               the API loads at startup
tests/
  test_api.py
```

## Dataset

[Amazon Books Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)
on Kaggle: `books_data.csv` (~212K books, metadata) joined with
`Books_rating.csv` (~3M reviews, ~2.8 GB) on `Title`. The full raw CSVs are
not committed to this repo — regenerate them locally with the steps below.
The committed `data/processed/` artifacts (~4 MB total) were built from the
**real** dataset: 25,380 unique books, sampled down from the full 3M reviews
the same way the notebook does (`sample(n=50000, random_state=42)`).

## Regenerating the artifacts from real data

```bash
pip install -r requirements-dev.txt   # adds kagglehub, needed only for download_data.py

# 1. Download the raw CSVs into data/raw/ (public dataset — no Kaggle
#    credentials were needed when this was last run, but if kagglehub asks,
#    set up ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY)
python scripts/download_data.py

# 2. Run the real pipeline and write data/processed/books_catalog.parquet,
#    data/processed/feature_matrix.npz, and data/processed/genre_keywords.json
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
textbox for sentiment analysis, a title-search box (backed by
`GET /books?q=`) for picking a book and viewing its recommendations, and a
genre search box (backed by `GET /genres` and `GET /genres/{genre}/keywords`)
showing praised vs. complained-about terms for that genre. It's just a thin
client over the existing JSON endpoints — open the browser devtools network
tab to see it calling them directly. CORS is wide open (`allow_origins=["*"]`)
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

curl -s "http://127.0.0.1:8000/recommend/B000IEZE3G?top_n=4"   # Harry Potter and the Sorcerer's Stone
# [
#   {"book_id":"B000NPEWHE","title":"Harry Potter and the Chamber of Secrets","genre":"Juvenile Fiction","similarity_score":0.8116},
#   {"book_id":"B000N76ZCC","title":"Harry Potter & the Prisoner of Azkaban","genre":"Literary Criticism","similarity_score":0.654},
#   {"book_id":"1553064690","title":"Elijah House and the Story of Chris Porter","genre":"Fiction","similarity_score":0.6193},
#   {"book_id":"1932586296","title":"The Toonies Invade Silicon Valley","genre":"Audiobooks","similarity_score":0.6043}
# ]

curl -s "http://127.0.0.1:8000/recommend/does-not-exist"
# {"detail":"book_id does-not-exist not found."}

curl -s http://127.0.0.1:8000/genres | python -m json.tool | head -5
# ["\"Childrens stories\"", "Adultery", "Adventure stories", "Africa", "African Americans", ...]

curl -s "http://127.0.0.1:8000/genres/Cooking/keywords"
# {
#   "genre": "Cooking",
#   "positive": ["cookbook","recipes","cooking","book","great","good","low carb","carb","excellent","great cookbook","easy","food"],
#   "negative": ["disappointed","disappointing","disappointed disappointing","jealous","diet","die","hurting","bbq","bland boring","book errors","cookbooks","cooking"]
# }
```

Note the Harry Potter result: the top two matches are correctly identified as
the next two books in the series *even though their genre labels in the raw
data are inconsistent* ("Juvenile Fiction" vs. "Literary Criticism") — the
content-based (TF-IDF) signal is carrying real weight here, not just genre.

(`book_id` is the Amazon `Id` field from the source data — an ASIN-like
string, not a small integer.)

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Covers: `/sentiment` on clearly positive/negative/empty input,
`/recommend/{book_id}` on a valid id (returns `top_n` results, none equal to
the input book) and an invalid id (404), `/books` search, `/genres` +
`/genres/{genre}/keywords`, and that the frontend is actually served at `/`.
All 11 tests pass against the real 25,380-book catalog.

## Notes on the recommender

The notebook builds a `Title x Genre` pivot table of rating-quality scores
and runs cosine similarity over it (`recommend_books_by_genre`, cell 36).
Reproduced as-is, that feature vector is one-hot-by-genre — each book has a
single nonzero entry, at its own genre column, holding its rating-quality
ordinal. Cosine similarity between two such vectors is *exactly* 1.0
whenever the books share a genre and 0.0 otherwise: the "recommender" is
really just a genre filter with no way to rank two same-genre books against
each other.

`app/recommender.py` fixes this by concatenating on a genuine content
signal — TF-IDF over each book's pooled review-summary text — alongside the
genre one-hot and a rating-quality scalar, each independently weighted, then
L2-normalized as one combined sparse vector per book. Two books in the same
genre now score anywhere in `(0, 1)` depending on how similar their reviews
actually read; the Harry Potter example above is the payoff — the two
sequels rank highest by review-text similarity even though their genre
labels in the raw data don't agree with each other.

It also pivots on `book_id` instead of `Title` (two different books can
share a title, and the API needs a stable per-book key), and never
materializes a dense N x N similarity matrix. At 25,380 books that matrix
would be ~4.8 GB of float64 — it doesn't fit in memory on a typical machine.
Because the *feature matrix* itself is sparse, that's what's precomputed and
persisted (`feature_matrix.npz`); cosine similarity for one queried book
against the whole catalog is a single O(n) sparse matrix-vector product,
computed on demand in `recommender.recommend_similar` rather than
precomputed for every pair up front.

**Two bugs surfaced while building this, worth naming honestly:**

1. The rating-quality scalar started out getting L2-normalized the same way
   as the other two blocks (`sklearn.preprocessing.normalize`). But
   normalizing a *single-column* vector by its own L2 norm always divides
   the value by itself — every book's rating column silently collapsed to a
   constant `1.0`, contributing nothing. Fixed by leaving that one block
   unnormalized (it's already scaled to `[0, 1]`) instead of running it
   through the same per-row L2 step as the genre and text blocks.
2. `TfidfVectorizer(max_features=N)` keeps the vocabulary's *N most frequent*
   terms and drops the rest — which is backwards for a discriminative
   signal: it's exactly the specific, distinguishing words (a character
   name, a niche topic) that are rare and get cut, while generic words
   ("book", "good", "great") survive. With a tight cap, most books' text
   vectors ended up all zero, collapsing them back onto identical
   genre+rating vectors — the same 1.0-tie problem, just disguised. Fixed
   with `min_df=2` (drop terms only one book uses — they can never
   contribute to *any* pairwise similarity anyway) and `max_df=0.5` (drop
   terms too common to discriminate) instead of a low `max_features` cap.

A residual, expected limitation: a book with only one or two reviews has too
little pooled text to produce a meaningful TF-IDF vector, so its
recommendations still fall back to genre + rating alone (verifiable: query
`/recommend/{book_id}` for any single-review book and compare to a
115-review book like `B000IEZE3G`). Using each review's full text
(`review/text`) instead of just its one-line summary would help, at the
cost of reading a much larger CSV column across 3M rows — a deliberate
trade against the memory constraints already documented in
`app/data_pipeline.py`.

## Genre Insights (keyword extraction)

`app/keywords.py` answers a different question than sentiment scoring or
recommendation: not "is this review positive," but "what specifically are
people praising or complaining about in this genre" — pattern discovery
from the review text itself, not just classification.

Approach: pool every positive review's summary text within a genre into one
document, and every negative review's summary text into another, then run
TF-IDF over the resulting corpus of per-(genre, sentiment) documents. TF-IDF's
IDF term does the real work here — words common across nearly every pool
("book", "read") get downweighted, while words distinctive to a specific
pool (`cookbook`, `recipes`, `low carb` for Cooking/positive; `christian`,
`faith`, `devotional` for Religion/positive) rise to the top. This was
chosen over LDA topic modeling specifically because it needs no
topic-count tuning or coherence evaluation to produce a usable, inspectable
result — see the `Cooking` example above. Genres with fewer than 20 reviews
of a given sentiment are skipped as too noisy to trust (`MIN_REVIEWS_PER_BUCKET`
in `app/keywords.py`).

Caveat worth naming: because `review/summary` is a short, one-line summary
rather than the full review body, a book's own title sometimes leaks into
its genre's keyword list (e.g. "Fiction/negative" surfacing `pride`,
`prejudice` from reviews of *Pride and Prejudice* that just restate the
title). This is genuine signal, not noise removal gone wrong — but it means
the keyword lists skew toward "what gets mentioned" more than a purer
topic-modeling approach would.

## Environment note

This was built and validated in a sandbox with no interactive Kaggle login;
`kagglehub.dataset_download` still succeeded because this dataset is public.
If you hit an auth prompt in a different environment, set up Kaggle API
credentials per https://github.com/Kaggle/kaggle-api#api-credentials.
