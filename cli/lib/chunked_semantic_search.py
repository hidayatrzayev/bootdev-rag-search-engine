import json
import os
import numpy as np

from lib.semantic_search import SemanticSearch

from utils import chunk_text
from chunking_strategy import ChunkBySentenceStrategy


class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self):
        super().__init__(embeddings_file_path="cache/chunk_embeddings.npy")
        self.chunk_metadata = None
        self.chunk_metadata_file_path = "cache/chunk_metadata.json"

    def _do_build(self, movies) -> np.ndarray:
        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for movie_index, movie in enumerate(movies):
            if not movie.description or movie.description.isspace():
                continue

            description_chunks = chunk_text(movie.description, chunk_size=4, overlap=1, chunking_strategy=ChunkBySentenceStrategy())
            all_chunks.extend(description_chunks)

            description_chunks_count = len(description_chunks)
            for chunk_index in range(description_chunks_count):
                chunk_metadata.append({
                    "movie_idx": movie_index,
                    "chunk_idx": chunk_index,
                    "total_chunks": description_chunks_count
                })

        self.chunk_metadata = chunk_metadata
        self.__save_chunk_metadata(total_chunks=len(all_chunks))

        return self.model.encode(all_chunks, show_progress_bar=True)

    def _embeddings_file_exists(self):
        return os.path.exists(self.embeddings_file_path) and os.path.exists(self.chunk_metadata_file_path)

    def _do_load_embeddings(self) -> np.ndarray:
        with open(self.chunk_metadata_file_path, "r") as f:
            self.chunk_metadata = json.load(f)

        with open(self.embeddings_file_path, "rb") as f:
            self.embeddings = np.load(f)
            return self.embeddings

    def __save_chunk_metadata(self, total_chunks: int):
        with open(self.chunk_metadata_file_path, "w") as f:
            json.dump({
                "chunks": self.chunk_metadata,
                "total_chunks": total_chunks,
            }, f, indent=2)
