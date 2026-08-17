from argparse import ArgumentParser

from cli.lib.semantic_search import verify_model


def parse_arguments(parser: ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify embeddings model information")
    return parser.parse_args()


def main() -> None:
    parser = ArgumentParser(description="Semantic Search CLI")
    args = parse_arguments(parser)

    match args.command:
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
