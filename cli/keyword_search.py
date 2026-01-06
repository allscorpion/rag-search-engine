from collections import Counter
import math
from constants import BM25_B, BM25_K1
from parse_tokens import parse_tokens

import pickle
from pathlib import Path


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = parse_tokens(text)
        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        total_docs = len(self.doc_lengths)
        total_length = sum(self.doc_lengths.values())

        return total_length / total_docs

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

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        tokens = parse_tokens(query)
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            sum = 0
            for token in tokens:
                bm25_score = self.bm25(doc_id, token)
                sum += bm25_score
            scores[doc_id] = sum
        sorted_scores = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )

        top_documents = []

        for doc_id in sorted_scores:
            sorted_score = sorted_scores[doc_id]
            doc = self.docmap[doc_id]
            top_documents.append((doc, sorted_score))
            if len(top_documents) >= limit:
                break
        return top_documents

    def save(self):
        Path("cache").mkdir(parents=True, exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)

        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open("cache/index.pkl", "rb") as f:
            self.index = pickle.load(f)
        with open("cache/docmap.pkl", "rb") as f:
            self.docmap = pickle.load(f)
        with open("cache/term_frequencies.pkl", "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open("cache/doc_lengths.pkl", "rb") as f:
            self.doc_lengths = pickle.load(f)
