import pickle

from text_processor import TextProcessor
from utils import load_movies


class InvertedIndex:
    __index: dict[str, set[int]] = {}
    __docmap: dict[int, dict] = {}
    __text_processor: TextProcessor = TextProcessor()

    def get_documents(self, term) -> list[int]:
        return sorted(self.__index.get(term, []))

    def get_movie_by_doc_id(self, doc_id) -> dict:
        return self.__docmap[doc_id]

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

        return self

    def load(self):
        with open("cache/index.pkl", "rb") as index_file:
            self.__index = pickle.load(index_file)

        with open("cache/docmap.pkl", "rb") as docmap_file:
            self.__docmap = pickle.load(docmap_file)

    def __add_document(self, doc_id: int, text: str):
        stemmed_tokens = self.__text_processor.process_text(text)
        for token in stemmed_tokens:
            if token in self.__index.keys():
                self.__index[token].add(doc_id)
            else:
                self.__index[token] = {doc_id}
