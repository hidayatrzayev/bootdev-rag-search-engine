import json

from movie import Movie
from chunking_strategy import ChunkingStrategy


def load_movies() -> list[Movie]:
    with open("data/movies.json") as f:
        return [Movie.from_dict(movie) for movie in json.load(f)["movies"]]


def chunk_text(
    text: str, 
    chunk_size: int, 
    overlap: int, 
    chunking_strategy: ChunkingStrategy,
) -> list[str]:
    print(f"Chunking {len(text)} characters")
    chunks = chunking_strategy.chunk(text)

    result: list[str] = []

    chunk_start_pos = 0
    while chunk_start_pos < len(chunks):
        first_chunk_pos = max(chunk_start_pos - overlap, 0)
        last_chunk_pos = min(first_chunk_pos + chunk_size, len(chunks))
        
        result.append(" ".join(chunks[first_chunk_pos:last_chunk_pos]))
        chunk_start_pos = first_chunk_pos + chunk_size

    return result
