from dataclasses import dataclass

@dataclass(eq=True, frozen=True)
class MovieSearchResult:
    score: float
    movie_title: str
    movie_description: str
