from pydantic import BaseModel


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    text: str
    compound_score: float
    sentiment: str


class RecommendationResponse(BaseModel):
    book_id: str
    title: str
    genre: str
    similarity_score: float


class BookSummary(BaseModel):
    book_id: str
    title: str
    genre: str


class GenreKeywords(BaseModel):
    genre: str
    positive: list[str]
    negative: list[str]


class HealthResponse(BaseModel):
    status: str
    books_loaded: int
    genres_with_keywords: int
