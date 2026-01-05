#!/usr/bin/env python3

import argparse
import json
import math
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


def handle_tf(document_id: int, term: str):
    try:
        cacher.load()
    except:
        print("please run build command before running this command")
        return
    tf = cacher.get_tf(document_id, term)
    print(f"{document_id} contains the term {term}, {tf} time(s)")


def get_idf(term: str):
    tokens = parse_tokens(term)

    if len(tokens) != 1:
        print("only provide a single token")

    token = tokens[0]

    total_doc_count = len(cacher.docmap)
    term_match_doc_count = len(cacher.get_documents(token))
    idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    return idf


def handle_idf(term: str):
    try:
        cacher.load()
    except:
        print("please run build command before running this command")
        return

    idf = get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def handle_tfidf(document_id: int, term: str):
    try:
        cacher.load()
    except:
        print("please run build command before running this command")
        return

    tf = cacher.get_tf(document_id, term)
    idf = get_idf(term)
    tf_idf = tf * idf

    print(f"TF-IDF score of '{term}' in document '{document_id}': {tf_idf:.2f}")


def handle_bm25_idf(term: str):
    try:
        cacher.load()
    except:
        print("please run build command before running this command")
        return
    bm25_idf = cacher.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movies into cache")
    term_frequency_parser = subparsers.add_parser(
        "tf", help="Check how frequent a term is"
    )
    term_frequency_parser.add_argument(
        "document_id", type=int, help="The document ID to check"
    )
    term_frequency_parser.add_argument("term", type=str, help="The term to lookup")

    idf_parser = subparsers.add_parser("idf", help="Check how common a term is")
    idf_parser.add_argument("term", type=str, help="The term to lookup")

    tf_idf_parser = subparsers.add_parser("tfidf", help="Check how common a term is")
    tf_idf_parser.add_argument("document_id", type=int, help="The document ID to check")
    tf_idf_parser.add_argument("term", type=str, help="The term to lookup")

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            handle_search(args.query)
        case "build":
            handle_build()
        case "tf":
            handle_tf(args.document_id, args.term)
        case "idf":
            handle_idf(args.term)
        case "tfidf":
            handle_tfidf(args.document_id, args.term)
        case "bm25idf":
            handle_bm25_idf(args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
