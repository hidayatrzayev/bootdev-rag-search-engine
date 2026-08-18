from argparse import ArgumentParser

from lib.semantic_search import verify_model, embed_text


def parse_arguments(parser: ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify embeddings model information")

    embed_parser = subparsers.add_parser("embed", help="Generate an embedding for the given text")
    embed_parser.add_argument("text", type=str, help="Text for which to generate an embedding")

    return parser.parse_args()


def main() -> None:
    parser = ArgumentParser(description="Semantic Search CLI")
    args = parse_arguments(parser)

    match args.command:
        case "verify":
            verify_model()
        case "embed":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
