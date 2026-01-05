from collections import Counter
import math
from parse_tokens import parse_tokens

import pickle
from pathlib import Path


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = parse_tokens(text)
        self.term_frequencies[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower(), set())
        return list(sorted(doc_ids))

    def build(self, movies):
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def parse_term_into_single_token(self, term):
        parsed_terms = parse_tokens(term)

        if len(parsed_terms) != 1:
            raise ValueError("term is invalid, please supply one valid term")

        parsed_term = parsed_terms[0]

        return parsed_term

    def get_tf(self, doc_id, term):
        token = self.parse_term_into_single_token(term)
        return self.term_frequencies[doc_id].get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
        token = self.parse_term_into_single_token(term)
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(token))
        bm25_idf = math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

        return bm25_idf

    def save(self):
        Path("cache").mkdir(parents=True, exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
