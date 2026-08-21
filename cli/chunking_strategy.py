import re

from abc import ABC, abstractmethod


class ChunkingStrategy(ABC):

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Chunk the given text and return the list of produced chunks"""


class ChunkByWordStrategy(ChunkingStrategy):

    def chunk(self, text: str) -> list[str]:
        return text.split()


class ChunkBySentenceStrategy(ChunkingStrategy):

    def chunk(self, text: str) -> list[str]:
        return re.split(r"(?<=[.!?])\s+", text)
