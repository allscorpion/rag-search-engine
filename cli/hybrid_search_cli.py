import argparse

from hybrid_search import normalize_scores, print_results, rrf_search, weighted_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize the keyword and semantic search scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", default=200, help="Scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Start a search that has weighted scores"
    )
    weighted_search_parser.add_argument("query", type=str, help="Text to query")
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="How weighted the query should be, a value between 0 and 1",
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many results to bring back",
    )

    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Start a search that has weighted scores"
    )
    rrf_search_parser.add_argument("query", type=str, help="Text to query")
    rrf_search_parser.add_argument(
        "-k",
        type=int,
        default=60,
        help="How weighted the query should be",
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many results to bring back",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Method to rerank",
    )
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Method to rerank",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            results = normalize_scores(args.scores)
            for result in results:
                print(f"* {result:.4f}")

        case "weighted-search":
            results = weighted_search(args.query, args.alpha, args.limit)

            for i, id in enumerate(results):
                result = results[id]
                print(f"{i + 1}. {result["title"]}")
                print(f"   Hybrid Score: {result["hybrid_score"]:.4f}")
                print(
                    f"   BM25: {result["keyword_score"]:.4f}, Semantic: {result["semantic_score"]:.4f}"
                )
                print(f"   {result["description"][:100]}")
        case "rrf-search":
            results = rrf_search(
                args.query,
                args.k,
                args.limit,
                args.enhance,
                args.rerank_method,
                args.evaluate,
            )

            print_results(results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
