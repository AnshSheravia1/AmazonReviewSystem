"""VADER sentiment wrapper.

Mirrors the analyzer usage in the notebook (cell 23): the same
SentimentIntensityAnalyzer, the same compound score, plus the notebook's
5-bucket quality label used for per-book aggregate scores.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def score_text(text: str) -> float:
    return _analyzer.polarity_scores(text)["compound"]


def label_from_compound(compound: float) -> str:
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def analyze(text: str) -> tuple[float, str]:
    compound = score_text(text)
    return compound, label_from_compound(compound)


def sentiment_category(compound: float) -> str:
    """Notebook's get_sentiment_category bucketing (cell 23), used for the
    per-book aggregate sentiment score in the processed catalog."""
    if compound <= -0.5:
        return "Poor"
    elif compound <= 0:
        return "Average"
    elif compound <= 0.5:
        return "Good"
    elif compound <= 0.75:
        return "Very Good"
    else:
        return "Must Read"
