import sys

from argparse import ArgumentParser, Namespace

from text_processor import TextProcessor
from inverted_index import InvertedIndex, BM25_DIMINISHING_RETURN


def parse_arguments(parser: ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build an inverted index for the movies dataset")

    tf_parser = subparsers.add_parser("tf", help="Calculate the term frequency for the given term in the document with the given ID")
    tf_parser.add_argument("doc_id", type=int, help="ID of the document in which to look for the term frequency")
    tf_parser.add_argument("term", type=str, help="Term whose frequency to calculate from the document")

    idf_parser = subparsers.add_parser("idf", help="Calculate the Inverse Document Frequency (IDF) for the given term")
    idf_parser.add_argument("term", type=str, help="Term whose IDF to calculate")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF (Term Frequency - Inverse Document Frequency) for the given term")
    tfidf_parser.add_argument("doc_id", type=int, help="ID of the document in which to look for the term frequency")
    tfidf_parser.add_argument("term", type=str, help="Term whose TF-IDF to calculate")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Calculate BM25 IDF score for the given term")
    bm25idf_parser.add_argument("term", type=str, help="Term whose BM25 IDF score to calculate")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Calculate BM25 TF score for the given term")
    bm25tf_parser.add_argument("doc_id", type=int, help="ID of the document in which to look for the term BM25 TF")
    bm25tf_parser.add_argument("term", type=str, help="Term whose BM25 TF score to calculate")
    bm25tf_parser.add_argument(
        "dr", 
        type=float, 
        nargs="?", 
        default=BM25_DIMINISHING_RETURN, 
        help="Tunable BM25 parameter to control diminishing return"
    )

    return parser.parse_args()


def load_inverted_index() -> InvertedIndex:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("No cached index file found - exiting program.")
        sys.exit(1)

    return inverted_index


def search_movie_by_query(query: str):
    print(f"Searching for: {query}")

    inverted_index = load_inverted_index()
    text_processor = TextProcessor()
    tokens = text_processor.process_text(query)

    search_results = []
    for token in tokens:
        documents = inverted_index.get_documents(token)
        for doc_id in documents:
            search_results.append(inverted_index.get_movie_by_doc_id(doc_id))
            if len(search_results) >= 5:
                break

        if len(search_results) >= 5:
            break  

    for matched_movie in search_results:
        print(f"{matched_movie["id"]}. {matched_movie["title"]}")


def build_inverted_index():
    inverted_index = InvertedIndex()
    inverted_index.build().save()


def calculate_and_print_tf(doc_id: int, term: str):
    term_frequency = load_inverted_index().get_tf(doc_id, term)
    print(f"Term frequency (TF) for term '{term}' in document with ID {doc_id} = {term_frequency}")


def calculate_and_print_idf(term: str):
    inverse_document_frequency = load_inverted_index().get_idf(term)
    print(f"Inverse document frequency of '{term}': {inverse_document_frequency:.2f}")


def calculate_and_print_tf_idf(doc_id: int, term: str):
    tf_idf = load_inverted_index().get_tf_idf(doc_id, term)
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")


def calculate_and_print_bm25_idf(term: str):
    bm25_idf = load_inverted_index().get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")


def calculate_and_print_bm25_tf(doc_id: int, term: str, diminishing_return: float):
    bm25_tf = load_inverted_index().get_bm25_tf(doc_id, term, diminishing_return)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25_tf:.2f}")


def main() -> None:
    parser = ArgumentParser(description="Keyword Search CLI")
    args = parse_arguments(parser)

    match args.command:
        case "search":
            search_movie_by_query(args.query)
        case "build":
            build_inverted_index()
        case "tf":
            calculate_and_print_tf(args.doc_id, args.term)
        case "idf":
            calculate_and_print_idf(args.term)
        case "tfidf":
            calculate_and_print_tf_idf(args.doc_id, args.term)
        case "bm25idf":
            calculate_and_print_bm25_idf(args.term)
        case "bm25tf":
            calculate_and_print_bm25_tf(args.doc_id, args.term, args.dr)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
