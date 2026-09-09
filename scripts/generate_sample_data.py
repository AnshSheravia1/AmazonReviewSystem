"""Generates a small synthetic dataset with the exact schema of the real
Kaggle "Amazon Books Reviews" CSVs (books_data.csv + Books_rating.csv).

This sandbox has no internet access / Kaggle credentials to pull the real
3-million-row dataset, so this script produces a schema-identical stand-in
that the *real* pipeline (app/data_pipeline.py) runs over unmodified, giving
genuine (non-hardcoded) precomputed artifacts to develop and test against.
Swap in the real CSVs (scripts/download_data.py) and rerun
scripts/build_artifacts.py for production.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_raw"

BOOKS = [
    ("Dune", "['Science Fiction']", "['Frank Herbert']"),
    ("Foundation", "['Science Fiction']", "['Isaac Asimov']"),
    ("Neuromancer", "['Science Fiction']", "['William Gibson']"),
    ("Hyperion", "['Science Fiction']", "['Dan Simmons']"),
    ("The Left Hand of Darkness", "['Science Fiction']", "['Ursula K. Le Guin']"),
    ("The Hobbit", "['Fantasy']", "['J.R.R. Tolkien']"),
    ("A Game of Thrones", "['Fantasy']", "['George R.R. Martin']"),
    ("The Name of the Wind", "['Fantasy']", "['Patrick Rothfuss']"),
    ("Mistborn", "['Fantasy']", "['Brandon Sanderson']"),
    ("The Way of Kings", "['Fantasy']", "['Brandon Sanderson']"),
    ("1984", "['Dystopian']", "['George Orwell']"),
    ("Brave New World", "['Dystopian']", "['Aldous Huxley']"),
    ("Fahrenheit 451", "['Dystopian']", "['Ray Bradbury']"),
    ("The Handmaid's Tale", "['Dystopian']", "['Margaret Atwood']"),
    ("We", "['Dystopian']", "['Yevgeny Zamyatin']"),
    ("Sapiens", "['Nonfiction']", "['Yuval Noah Harari']"),
    ("Educated", "['Nonfiction']", "['Tara Westover']"),
    ("Into the Wild", "['Nonfiction']", "['Jon Krakauer']"),
    ("Quiet", "['Nonfiction']", "['Susan Cain']"),
    ("Outliers", "['Nonfiction']", "['Malcolm Gladwell']"),
    ("Gone Girl", "['Mystery']", "['Gillian Flynn']"),
    ("The Girl with the Dragon Tattoo", "['Mystery']", "['Stieg Larsson']"),
    ("And Then There Were None", "['Mystery']", "['Agatha Christie']"),
    ("The Silent Patient", "['Mystery']", "['Alex Michaelides']"),
    ("Big Little Lies", "['Mystery']", "['Liane Moriarty']"),
    ("Pride and Prejudice", "['Unknown']", "['Jane Austen']"),
    ("Sense and Sensibility", "['Unknown']", "['Jane Austen']"),
    ("Emma", "['Unknown']", "['Jane Austen']"),
    ("Untitled Manuscript", "['Unknown']", "['Unknown']"),
    ("Forgotten Pages", "['Unknown']", "['Unknown']"),
]

POSITIVE_SUMMARIES = [
    "Absolutely loved this book, could not put it down",
    "A masterpiece, brilliant writing and unforgettable characters",
    "Best book I've read all year, highly recommend",
    "Wonderful story, beautifully written and deeply moving",
    "Fantastic read, exceeded all my expectations",
]
NEGATIVE_SUMMARIES = [
    "Terrible book, complete waste of time",
    "I hated this, boring and poorly written",
    "Disappointing and dull, would not recommend",
    "Awful pacing and flat characters, a real letdown",
    "One of the worst books I've ever read",
]
NEUTRAL_SUMMARIES = [
    "It was okay, nothing special",
    "An average read, some good parts some slow parts",
    "Decent book but forgettable",
]

FIRST_NAMES = ["Alex", "Jamie", "Taylor", "Morgan", "Casey", "Jordan", "Riley", "Sam"]


def make_review_id(i: int) -> str:
    return f"B{i:09d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_RAW_DIR))
    args = parser.parse_args()

    RAW_DIR = Path(args.output_dir)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    book_data_rows = []
    rating_rows = []

    for i, (title, categories, authors) in enumerate(BOOKS):
        book_id = make_review_id(i)
        book_data_rows.append(
            {
                "Title": title,
                "description": f"A description of {title}." if rng.random() > 0.1 else None,
                "authors": authors,
                "image": None,
                "previewLink": None,
                "publisher": rng.choice(["Ace Books", "Tor", "Penguin", None]),
                "publishedDate": str(1950 + rng.randint(0, 74)),
                "infoLink": None,
                "categories": categories,
            }
        )

        n_reviews = rng.randint(5, 15)
        for _ in range(n_reviews):
            score = rng.choices([1.0, 2.0, 3.0, 4.0, 5.0], weights=[1, 1, 2, 3, 3])[0]
            if score >= 4:
                summary = rng.choice(POSITIVE_SUMMARIES)
            elif score <= 2:
                summary = rng.choice(NEGATIVE_SUMMARIES)
            else:
                summary = rng.choice(NEUTRAL_SUMMARIES)

            rating_rows.append(
                {
                    "Id": book_id,
                    "Title": title,
                    "Price": rng.choice([None, 9.99, 14.99, 19.99]),
                    "User_id": f"U{rng.randint(10**8, 10**9 - 1)}",
                    "profileName": rng.choice(FIRST_NAMES),
                    "review/helpfulness": f"{rng.randint(0, 10)}/{rng.randint(0, 10)}",
                    "review/score": score,
                    "review/time": 1000000000,
                    "review/summary": summary,
                    "review/text": summary + ". " + summary,
                }
            )

    pd.DataFrame(book_data_rows).to_csv(RAW_DIR / "books_data.csv", index=False)
    pd.DataFrame(rating_rows).to_csv(RAW_DIR / "Books_rating.csv", index=False)
    print(
        f"Wrote {len(book_data_rows)} books and {len(rating_rows)} reviews to "
        f"{RAW_DIR}/books_data.csv and {RAW_DIR}/Books_rating.csv"
    )


if __name__ == "__main__":
    main()
