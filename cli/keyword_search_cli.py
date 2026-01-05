#!/usr/bin/env python3

import argparse
import json
from parse_tokens import parse_tokens, contains_token
from movie_cache import InvertedIndex


def get_movies():
    with open("data/movies.json") as f:
        data = json.load(f)
        return data["movies"]


cacher = InvertedIndex()


def get_active_doc_ids(tokens):
    active_doc_ids: set[int] = set()
    for token in tokens:
        doc_ids = cacher.get_documents(token)
        for doc_id in doc_ids:
            if len(active_doc_ids) >= 5:
                return active_doc_ids
            active_doc_ids.add(doc_id)

    return active_doc_ids


def handle_search(search):
    try:
        cacher.load()
    except Exception as e:
        print("please run build command before trying to search")
        return
    print(f"Searching for: {search}")
    search_tokens = parse_tokens(search)
    active_doc_ids = get_active_doc_ids(search_tokens)

    for doc_id in active_doc_ids:
        document = cacher.docmap[doc_id]
        print(f"{doc_id}. Movie Title {document['title']}")


def handle_build():
    cacher.build(get_movies())
    cacher.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movies into cache")

    args = parser.parse_args()

    match args.command:
        case "search":
            handle_search(args.query)
        case "build":
            handle_build()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
