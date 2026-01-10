import argparse

from hybrid_search import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize the keyword and semantic search scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", default=200, help="Scores to normalize"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
