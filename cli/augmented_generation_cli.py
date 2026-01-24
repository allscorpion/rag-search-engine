import argparse

from augmented_generation import (
    citations_search,
    question_search,
    rag_search,
    summarize_search,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + generate answer)"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Search query for RAG"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + generate answer + citations)"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Search query for RAG"
    )

    question_parser = subparsers.add_parser(
        "question", help="Perform RAG (search + generate answer + question)"
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Search query for RAG"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_search(query)
        case "summarize":
            summarize_search(args.query, args.limit)
        case "citations":
            citations_search(args.query, args.limit)
        case "question":
            question_search(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
