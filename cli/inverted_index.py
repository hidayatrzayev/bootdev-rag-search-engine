import pickle
import math
import os

from collections import Counter

from text_processor import TextProcessor
from utils import load_movies
from movie import Movie


LAPLACE_SMOOTHING = 0.5
BM25_DIMINISHING_RETURN = 1.5
BM25_DOC_LENGTH_NORMALIZATION_STRENGTH = 0.75
BM25_DEFAULT_SEARCH_LIMIT = 5
CACHE_DIR = "cache"


class InvertedIndex:
    __index: dict[str, set[int]] = {}
    __docmap: dict[int, Movie] = {}
    __term_frequencies: dict[int, Counter] = {}
    __doc_lengths: dict[int, int] = {}

    __text_processor: TextProcessor = TextProcessor()

    def get_documents(self, term) -> list[int]:
        return sorted(self.__index.get(term, []))

    def get_movie_by_doc_id(self, doc_id) -> Movie:
        return self.__docmap[doc_id]

    def get_tf(self, doc_id: int, term: str) -> int:
        doc_counter = self.__term_frequencies.get(doc_id)
        if doc_counter is None:
            return 0

        return doc_counter[self.__ensure_single_token(term)]

    def get_bm25_tf(
            self, 
            doc_id: int, 
            term: str, 
            diminishing_return: float = BM25_DIMINISHING_RETURN,
            norm_strength: float = BM25_DOC_LENGTH_NORMALIZATION_STRENGTH,
        ) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - norm_strength + norm_strength * (self.__doc_lengths[doc_id] / self.__average_doc_length())

        return (tf * (diminishing_return + 1)) / (tf + diminishing_return * length_norm)

    def get_idf(self, term: str) -> float:
        token = self.__ensure_single_token(term)
        return math.log((self.__total_document_count() + 1) / (self.__document_frequency(token) + 1))

    def get_bm25_idf(self, term: str) -> float:
        df = self.__document_frequency(self.__ensure_single_token(term))
        return math.log((self.__total_document_count() - df + LAPLACE_SMOOTHING) / (df + LAPLACE_SMOOTHING) + 1)

    def get_tf_idf(self, doc_id: str, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = BM25_DEFAULT_SEARCH_LIMIT) -> dict[Movie, float]:
        scores: dict[int, float] = {}
        tokens = self.__text_processor.process_text(query)

        for token in tokens:
            for doc_id in self.__index.get(token, {}):
                if doc_id in scores.keys():
                    scores[doc_id] += self.bm25(doc_id, token)
                else:
                    scores[doc_id] = self.bm25(doc_id, token)

        return {
            self.get_movie_by_doc_id(doc_id): score 
            for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        }

    def build(self):
        for movie in load_movies():
            self.__add_document(movie.id, f"{movie.title} {movie.description}")
            self.__docmap[movie.id] = movie

        return self

    def save(self):
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as index_file:
            pickle.dump(self.__index, index_file)

        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as docmap_file:
            pickle.dump(self.__docmap, docmap_file)

        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as tf_file:
            pickle.dump(self.__term_frequencies, tf_file)

        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), "wb") as dl_file:
            pickle.dump(self.__doc_lengths, dl_file)

        return self

    def load(self):
        with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as index_file:
            self.__index = pickle.load(index_file)

        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as docmap_file:
            self.__docmap = pickle.load(docmap_file)

        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as tf_file:
            self.__term_frequencies = pickle.load(tf_file)

        with open(os.path.join(CACHE_DIR, "doc_lengths.pkl"), "rb") as dl_file:
            self.__doc_lengths = pickle.load(dl_file)

    def __add_document(self, doc_id: int, text: str):
        stemmed_tokens = self.__text_processor.process_text(text)
        for token in stemmed_tokens:
            if token in self.__index.keys():
                self.__index[token].add(doc_id)
            else:
                self.__index[token] = {doc_id}

            if doc_id in self.__term_frequencies.keys():
                self.__term_frequencies[doc_id].update({token})
            else:
                self.__term_frequencies[doc_id] = Counter({token})

        if doc_id in self.__doc_lengths.keys():
            self.__doc_lengths[doc_id] += len(stemmed_tokens)
        else:
            self.__doc_lengths[doc_id] = len(stemmed_tokens)

    def __total_document_count(self):
        return len(self.__docmap)

    def __document_frequency(self, term: str) -> int:
        return len(self.get_documents(term))

    def __average_doc_length(self) -> float:
        total_doc_count = len(self.__doc_lengths)
        if total_doc_count == 0:
            return 0.0
        
        return sum(self.__doc_lengths.values()) / total_doc_count

    def __ensure_single_token(self, term: str) -> str:
        tokenized = self.__text_processor.process_text(term)

        if len(tokenized) > 1:
            raise ValueError(f"Expected to tokenize term {term} to exactly 1 token, but was {len(tokenized)}")

        return tokenized[0]
