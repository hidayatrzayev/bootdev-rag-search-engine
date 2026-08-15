import string

from nltk.stem import PorterStemmer

punctuation_removal_table = str.maketrans({
    punctuation: None for punctuation in string.punctuation
})


class TextProcessor:
    _stop_words: list[str]
    _stemmer: PorterStemmer

    def __init__(self):
        with open("data/stopwords.txt", "r") as f:
            self._stop_words = [self._remove_punctuation(sw) for sw in f.read().splitlines()]
            self._stemmer = PorterStemmer()

    def process_text(self, text: str) -> list[str]:
        tokens = self._tokenize_by_word(self._remove_punctuation(text.lower()))
        return [self._stemmer.stem(token) for token in tokens]

    def _remove_punctuation(self, text: str) -> str:
        return text.translate(punctuation_removal_table)

    def _is_stop_word(self, token: str) -> bool:
        return token in self._stop_words

    def _is_valid_token(self, token: str) -> bool:
        return not token.isspace() and not self._is_stop_word(token)

    def _tokenize_by_word(self, text: str) -> list[str]:
        return filter(lambda token: self._is_valid_token(token), text.split())
