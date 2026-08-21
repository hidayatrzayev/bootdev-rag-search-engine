from argparse import ArgumentParser

from lib.semantic_search import verify_model, embed_text, verify_embeddings, SemanticSearch
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.movie_search_result import MovieSearchResult
from utils import load_movies, chunk_text
from chunking_strategy import ChunkByWordStrategy, ChunkBySentenceStrategy


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

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text into smaller pieces for embedding")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=200,
        help="Optional parameter to specify a chunk size"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Optional parameter to specify the amount of overlapping words in chunks"
    )

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a given text into semantically integral parts (e.g. sentences) for embedding")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=4,
        help="Optional parameter to specify a maximum chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=0,
        help="Optional parameter to specify the amount of overlapping parts in chunks"
    )

    subparsers.add_parser("embed_chunks", help="Generate chunked embeddings for the movie dataset")

    return parser.parse_args()


def search_movies(query: str, limit: int) -> list[MovieSearchResult]:
    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(load_movies())
    return semantic_search.search(query, limit)


def print_results(results: list[MovieSearchResult]) -> None:
    for index, result in enumerate(results):
        print(f"{index + 1}. {result.movie_title} (score: {result.score})")
        print(f"  {result.movie_description[:MAX_PRINTED_CHARS]}{" ..." if len(result.movie_description) > MAX_PRINTED_CHARS else "."}\n")


def print_chunks(chunks: list[str]) -> None:
    for index, chunk in enumerate(chunks):
        print(f"{index + 1}. {chunk}")


def embed_movie_chunks():
    movies = load_movies()
    semantic_search = ChunkedSemanticSearch()
    return semantic_search.load_or_create_embeddings(movies)


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
        case "chunk":
            print_chunks(chunk_text(args.text, args.chunk_size, args.overlap, ChunkByWordStrategy()))
        case "semantic_chunk":
            print_chunks(
                chunk_text(
                    text=args.text, 
                    chunk_size=args.max_chunk_size, 
                    overlap=args.overlap, 
                    chunking_strategy=ChunkBySentenceStrategy(),
                    chunk_size_overflow_allowed=False
                )
            )
        case "embed_chunks":
            embeddings = embed_movie_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
