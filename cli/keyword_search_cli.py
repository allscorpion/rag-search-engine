#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def get_stop_words():
    with open("data/stopwords.txt") as f:
        text = f.read()
        lines = text.splitlines()
        return lines


def convert_string_to_tokens(str):
    tokens = str.lower().translate(str.maketrans("", "", string.punctuation)).split()
    parsedTokens = list(filter(lambda token: token != "", tokens))
    return parsedTokens


def containsToken(tokenSet1, tokenSet2):
    for item in tokenSet1:
        for item2 in tokenSet2:
            if item in item2:
                return True
    return False


def filter_out_stop_words(stop_words, tokens):
    result = []
    for token in tokens:
        if token not in stop_words:
            result.append(token)

    return result


def stem_tokens(tokens):
    result = []
    for token in tokens:
        result.append(stemmer.stem(token))

    return result


def parse_tokens(str, stop_words):
    tokens = convert_string_to_tokens(str)
    filtered_out_stop_words = filter_out_stop_words(stop_words, tokens)
    stemmed_tokens = stem_tokens(filtered_out_stop_words)

    return stemmed_tokens


def handle_search(search):
    print(f"Searching for: {search}")
    stop_words = get_stop_words()
    with open("data/movies.json") as f:
        data = json.load(f)
        result = []
        search_tokens = parse_tokens(search, stop_words)
        for movie in data["movies"]:
            movie_tokens = parse_tokens(movie["title"], stop_words)
            if containsToken(search_tokens, movie_tokens):
                result.append(movie)

        resultParsed = sorted(result, key=lambda item: item["id"])[:5]

        for i in range(len(resultParsed)):
            item = resultParsed[i]
            print(f"{i + 1}. Movie Title {item['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            handle_search(args.query)
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
