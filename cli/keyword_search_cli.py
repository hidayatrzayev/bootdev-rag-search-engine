import sys
import math

from argparse import ArgumentParser

from text_processor import TextProcessor
from inverted_index import InvertedIndex
from utils import load_movies


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

    tf_idf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF (Term Frequency - Inverse Document Frequency) for the given term")
    tf_idf_parser.add_argument("doc_id", type=int, help="ID of the document in which to look for the term frequency")
    tf_idf_parser.add_argument("term", type=str, help="Term whose TF-IDF to calculate")

    return parser.parse_args()


def load_inverted_index() -> InvertedIndex:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("No cached index file found - exiting program.")
        sys.exit(1)

    return inverted_index


def tokenize_single_term(term: str) -> str:
    text_processor = TextProcessor()
    token = text_processor.process_text(term)
    if len(token) > 1:
        raise ValueError(f"Expected to tokenize term {term} to exactly 1 token, but was {len(token)}")

    return token[0]


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


def calculate_tf_in_document(doc_id: int, term: str, index: InvertedIndex) -> int:
    return index.get_tf(doc_id, term)


def calculate_idf_score(term: str, index: InvertedIndex) -> float:
    return math.log((index.get_total_document_count() + 1) / (index.get_matching_document_count(term) + 1))


def calculate_tf_idf_score(doc_id: int, term: str, index: InvertedIndex) -> float:
    return calculate_tf_in_document(doc_id, term, index) * calculate_idf_score(term, index)


def main() -> None:
    parser = ArgumentParser(description="Keyword Search CLI")
    args = parse_arguments(parser)

    match args.command:
        case "search":
            search_movie_by_query(args.query)
        case "build":
            build_inverted_index()
        case "tf":
            term_frequency = calculate_tf_in_document(
                args.doc_id, 
                term=tokenize_single_term(args.term), 
                index=load_inverted_index()
            )
            print(f"Term frequency (TF) for term '{args.term}' in document with ID {args.doc_id} = {term_frequency}")
        case "idf":
            inverse_document_frequency = calculate_idf_score(
                term=tokenize_single_term(args.term), 
                index=load_inverted_index()
            )
            print(f"Inverse document frequency of '{args.term}': {inverse_document_frequency:.2f}")
        case "tfidf":
            tf_idf = calculate_tf_idf_score(
                args.doc_id, 
                term=tokenize_single_term(args.term),
                index=load_inverted_index()
            )
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
