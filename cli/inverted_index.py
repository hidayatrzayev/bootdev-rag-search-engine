import pickle
import math

from collections import Counter

from text_processor import TextProcessor
from utils import load_movies


class InvertedIndex:
    __index: dict[str, set[int]] = {}
    __docmap: dict[int, dict] = {}
    __term_frequencies: dict[int, Counter] = {}

    __text_processor: TextProcessor = TextProcessor()

    def get_documents(self, term) -> list[int]:
        return sorted(self.__index.get(term, []))

    def get_movie_by_doc_id(self, doc_id) -> dict:
        return self.__docmap[doc_id]

    def get_tf(self, doc_id: int, term: str) -> int:
        doc_counter = self.__term_frequencies.get(doc_id)
        if doc_counter is None:
            return 0

        return doc_counter[term]

    def get_idf(self, term: str) -> float:
        return math.log((len(self.__docmap) + 1) / (len(self.get_documents(term)) + 1))

    def get_tf_idf(self, doc_id: str, term: str) -> float:
        return self.get_tf(doc_id, term) * self.get_idf(term)

    def build(self):
        for movie in load_movies():
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.__docmap[movie["id"]] = movie

        return self

    def save(self):
        with open("cache/index.pkl", "wb") as index_file:
            pickle.dump(self.__index, index_file)

        with open("cache/docmap.pkl", "wb") as docmap_file:
            pickle.dump(self.__docmap, docmap_file)

        with open("cache/term_frequencies.pkl", "wb") as tf_file:
            pickle.dump(self.__term_frequencies, tf_file)

        return self

    def load(self):
        with open("cache/index.pkl", "rb") as index_file:
            self.__index = pickle.load(index_file)

        with open("cache/docmap.pkl", "rb") as docmap_file:
            self.__docmap = pickle.load(docmap_file)

        with open("cache/term_frequencies.pkl", "rb") as tf_file:
            self.__term_frequencies = pickle.load(tf_file)

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
