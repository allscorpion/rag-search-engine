#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    chunk_text,
    embed_chunks,
    embed_query_text,
    search_documents,
    semantic_chunk_text,
    verify_embeddings,
    verify_model,
    embed_text,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify that the model is loaded")
    subparsers.add_parser(
        "verify_embeddings", help="Verify that the embeddings are working"
    )

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate text embedding"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to query")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate text embedding"
    )
    embed_query_parser.add_argument("query", type=str, help="Text to query")

    search_parser = subparsers.add_parser(
        "search", help="Search for the closest results"
    )
    search_parser.add_argument("query", type=str, help="Text to query")
    search_parser.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Top x results to show"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk the text into groups")
    chunk_parser.add_argument("text", type=str, help="Text to query")
    chunk_parser.add_argument(
        "--chunk-size", type=int, nargs="?", default=200, help="Chunk size"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="Overlap amount"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk the text into groups"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to query")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, nargs="?", default=4, help="Max chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="Overlap amount"
    )

    subparsers.add_parser("embed_chunks", help="Embed the documents into chunks")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_documents(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
