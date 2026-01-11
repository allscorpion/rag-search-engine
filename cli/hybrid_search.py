import os

from utils import get_movies
from keyword_search import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists("cache/index.pkl"):
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit=5):
        keyword_documents = self._bm25_search(query, 500 * limit)
        semantic_documents = self.semantic_search.search_chunks(query, 500 * limit)

        b25_scores = []
        for keyword_document in keyword_documents:
            b25_scores.append(
                {
                    "doc_id": keyword_document[0]["id"],
                    "score": keyword_document[1],
                }
            )
        normalized_b25_scores = normalize_search_results(b25_scores)

        semantic_scores = []
        for semantic_document in semantic_documents:
            semantic_scores.append(
                {
                    "doc_id": semantic_document["id"],
                    "score": semantic_document["score"],
                }
            )
        normalized_semantic_scores = normalize_search_results(semantic_scores)

        results: dict[int, dict] = {}

        for document in self.documents:
            b25_item = next(
                item
                for item in normalized_b25_scores
                if item["doc_id"] == document["id"]
            )
            semantic_item = next(
                item
                for item in normalized_semantic_scores
                if item["doc_id"] == document["id"]
            )
            b25_score = b25_item["normalized_score"]
            semantic_score = semantic_item["normalized_score"]

            results[document["id"]] = {
                "id": document["id"],
                "title": document["title"],
                "description": document["description"],
                "keyword_score": b25_score,
                "semantic_score": semantic_score,
                "hybrid_score": hybrid_score(b25_score, semantic_score, alpha),
            }

        sorted_results = dict(
            sorted(
                results.items(), key=lambda item: item[1]["hybrid_score"], reverse=True
            )[:limit]
        )
        return sorted_results

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores: list[float] = []
    for result in results:
        scores.append(result["score"])

    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def normalize_scores(scores: list[float]):
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    results = []
    for s in scores:
        results.append((s - min_score) / (max_score - min_score))

    return results


def weighted_search(query: str, alpha: float, limit: int):
    documents = get_movies()
    hybrid_search = HybridSearch(documents)
    return hybrid_search.weighted_search(query, alpha, limit)
