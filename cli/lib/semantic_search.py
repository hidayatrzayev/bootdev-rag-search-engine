import numpy as np
import os

from sentence_transformers import SentenceTransformer
from torch import Tensor

from movie import Movie
from lib.movie_search_result import MovieSearchResult
from utils import load_movies
from lib.vector import cosine_similarity

EMBEDDINGS_FILE_PATH = "cache/movie_embeddings.npy"

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: Tensor = None
        self.documents: list[Movie] = None
        self.document_map: dict[int, Movie] = None

    def generate_embedding(self, text: str):
        if not text or text.isspace():
            raise ValueError(f"Cannot generate an embedding for an empty text: {text}")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Movie]) -> Tensor:
        self.__populate_document_data(documents)

        movies = [f"{movie.title}: {movie.description}" for movie in documents]
        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        self.__save_embeddings()

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.__populate_document_data(documents)

        if os.path.exists(EMBEDDINGS_FILE_PATH):
            print(f"Embeddings already exist - loading from cache")
            with open(EMBEDDINGS_FILE_PATH, "rb") as f:
                self.embeddings = np.load(f)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        else:
            print(f"Embeddings do not yet exist - building from scratch")
            return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[MovieSearchResult]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)
        cosine_similarities = [(
            cosine_similarity(query_embedding, self.embeddings[i]), 
            self.documents[i]
        ) for i in range(len(self.embeddings))]

        return [
            MovieSearchResult(score, movie.title, movie.description)
            for score, movie in sorted(cosine_similarities, key=lambda x: x[0], reverse=True)[:limit]
        ]

    def __populate_document_data(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {movie.id: movie for movie in documents}

    def __save_embeddings(self):
        with open(EMBEDDINGS_FILE_PATH, "wb") as f:
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
