from argparse import ArgumentParser

from lib.semantic_search import verify_model, embed_text, verify_embeddings, SemanticSearch
from lib.movie_search_result import MovieSearchResult
from utils import load_movies


MAX_PRINTED_CHARS = 50


def parse_arguments(parser: ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify embeddings model information")

    embed_parser = subparsers.add_parser("embed", help="Generate an embedding for the given text")
    embed_parser.add_argument("text", type=str, help="Text for which to generate an embedding")

    subparsers.add_parser("verify_embeddings", help="Verify vector embeddings of the movie dataset")

    search_parser = subparsers.add_parser("search", help="Perform semantic search on a movie dataset")
    search_parser.add_argument("query", type=str, help="Query to perform semantic search on")
    search_parser.add_argument(
        "limit", 
        type=int, 
        nargs="?", 
        default=5, 
        help="Optional limit of how many results to print"
    )

    return parser.parse_args()


def search_movies(query: str, limit: int) -> list[MovieSearchResult]:
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(load_movies())
    return semantic_search.search(query, limit)


def print_results(results: list[MovieSearchResult]) -> None:
    for index, result in enumerate(results):
        print(f"{index + 1}. {result.movie_title} (score: {result.score})")
        print(f"  {result.movie_description[:MAX_PRINTED_CHARS]}{" ..." if len(result.movie_description) > MAX_PRINTED_CHARS else "."}\n")


def main() -> None:
    parser = ArgumentParser(description="Semantic Search CLI")
    args = parse_arguments(parser)

    match args.command:
        case "verify":
            verify_model()
        case "embed":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            print_results(search_movies(args.query, args.limit))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
