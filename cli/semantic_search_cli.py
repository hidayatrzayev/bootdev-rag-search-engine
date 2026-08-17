from argparse import ArgumentParser


def parse_arguments(parser: ArgumentParser):
    return parser.parse_args()


def main() -> None:
    parser = ArgumentParser(description="Semantic Search CLI")
    args = parse_arguments(parser)

    match args.command:
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
