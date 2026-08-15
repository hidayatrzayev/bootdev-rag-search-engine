from text_processor import TextProcessor


class KeywordMatcher:

    def __init__(self):
        self._text_processor = TextProcessor()

    def match_found(self, query: str, title: str) -> bool:
        query_tokens = self._text_processor.process_text(query)
        title_tokens = self._text_processor.process_text(title)

        for query_token in query_tokens:
            for title_token in title_tokens:
                if query_token in title_token:
                    return True

        return False
