import argparse
import json


from keyword_matcher import KeywordMatcher


def load_movies() -> dict:
    with open("data/movies.json") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    movies = load_movies()["movies"]

    search_results = []
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            keyword_matcher = KeywordMatcher()

            for movie in movies:
                title = movie["title"]

                if keyword_matcher.match_found(args.query, title):
                    search_results.append(title)

            for index, title in enumerate(search_results[:5]):
                print(f"{index + 1}. {title}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
