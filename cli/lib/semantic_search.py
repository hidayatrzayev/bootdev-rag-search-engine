import numpy as np
import os

from sentence_transformers import SentenceTransformer

from movie import Movie
from lib.movie_search_result import MovieSearchResult
from utils import load_movies
from lib.vector import cosine_similarity


class SemanticSearch:
    def __init__(self, embeddings_file_path: str = "cache/movie_embeddings.npy"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: np.ndarray = None
        self.documents: list[Movie] = None
        self.document_map: dict[int, Movie] = None
        self.embeddings_file_path = embeddings_file_path

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or text.isspace():
            raise ValueError(f"Cannot generate an embedding for an empty text: {text}")

        return self.model.encode([text])[0]

    def build_embeddings(self, movies: list[Movie]) -> np.ndarray:
        self.__populate_document_data(movies)

        self.embeddings = self._do_build(movies)
        self.__save_embeddings()

        return self.embeddings

    def load_or_create_embeddings(self, movies: list[Movie]):
        self.__populate_document_data(movies)

        if self._embeddings_file_exists():
            print(f"Embeddings already exist - loading from cache")
            return self._do_load_embeddings()
        else:
            print(f"Embeddings do not yet exist - building from scratch")
            return self.build_embeddings(movies)

    def search(self, query: str, limit: int) -> list[MovieSearchResult]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        return self._do_search(self.generate_embedding(query), limit)

    def _do_build(self, movies: list[Movie]) -> np.ndarray:
        movies_info = [f"{movie.title}: {movie.description}" for movie in movies]
        return self.model.encode(movies_info, show_progress_bar=True)

    def _embeddings_file_exists(self) -> bool:
        return os.path.exists(self.embeddings_file_path)

    def _do_load_embeddings(self) -> np.ndarray:
        with open(self.embeddings_file_path, "rb") as f:
            self.embeddings = np.load(f)
        
            if len(self.embeddings) == len(self.documents):
                return self.embeddings

    def _do_search(self, query_embedding: np.ndarray, limit: int) -> list[MovieSearchResult]:
        similarity_scores = [(
            cosine_similarity(query_embedding, self.embeddings[i]), 
            self.documents[i]
        ) for i in range(len(self.embeddings))]

        return [
            MovieSearchResult(score, movie.title, movie.description[:100])
            for score, movie in sorted(similarity_scores, key=lambda x: x[0], reverse=True)[:limit]
        ]

    def __populate_document_data(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {movie.id: movie for movie in documents}

    def __save_embeddings(self):
        with open(self.embeddings_file_path, "wb") as f:
            np.save(f, self.embeddings)


def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)

    print(f"Number of movie documents: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
