import json
import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np

from utils import get_movies

cache_movie_embeddings_path = "cache/movie_embeddings.npy"


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text.strip() == "":
            raise ValueError("invalid text")

        return self.model.encode([text])[0]

    def store_documents(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

    def build_embeddings(self, documents):
        self.store_documents(documents)
        document_strs = []
        for document in documents:
            document_strs.append(f"{document["title"]}: {document["description"]}")

        self.embeddings = self.model.encode(document_strs, show_progress_bar=True)
        np.save(cache_movie_embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.store_documents(documents)

        if os.path.exists(cache_movie_embeddings_path):
            self.embeddings = np.load(cache_movie_embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit=5):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        embedding = self.generate_embedding(query)
        results = []
        for i in range(len(self.embeddings)):
            document_embedding = self.embeddings[i]
            document = self.documents[i]
            similarity_score = cosine_similarity(embedding, document_embedding)
            results.append((similarity_score, document))

        sorted_results = sorted(results, key=lambda item: item[0], reverse=True)
        output = []
        for sorted_result in sorted_results[:limit]:
            output.append(
                {
                    "score": sorted_result[0],
                    "title": sorted_result[1]["title"],
                    "description": sorted_result[1]["description"],
                }
            )
        return output


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.store_documents(documents)
        chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for document_index, document in enumerate(documents):
            if not document["description"]:
                continue
            document_chunks = semantic_chunk(document["description"], 4, 1)
            total_chunks = len(document_chunks)
            for chunk_index in range(total_chunks):
                chunks.append(document_chunks[chunk_index])
                chunk_metadata.append(
                    {
                        "movie_idx": document_index,
                        "chunk_idx": chunk_index,
                        "total_chunks": total_chunks,
                    }
                )
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(chunk_metadata)},
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.store_documents(documents)
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists(
            "cache/chunk_metadata.json"
        ):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        if self.chunk_metadata is None or len(self.chunk_metadata) == 0:
            raise ValueError(
                "No chunk_metadata loaded. Call `load_or_create_embeddings` first."
            )

        embedding = self.generate_embedding(query)
        chunk_scores = []
        for i in range(len(self.chunk_embeddings)):
            document_embedding = self.chunk_embeddings[i]
            chunk_metadata = self.chunk_metadata[i]
            similarity_score = cosine_similarity(embedding, document_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": chunk_metadata["movie_idx"],
                    "score": similarity_score,
                }
            )
        movie_scores: dict[int, int] = {}
        for chunk_score in chunk_scores:
            movie_index = chunk_score["movie_idx"]
            if movie_index not in movie_scores:
                movie_scores[movie_index] = 0

            movie_scores[movie_index] = max(
                movie_scores[movie_index], chunk_score["score"]
            )

        sorted_movie_scores = dict(
            sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
        )
        results = []
        for movie_index in sorted_movie_scores:
            document = self.documents[movie_index]
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "description": document["description"][:100],
                    "score": round(sorted_movie_scores[movie_index], 2),
                    "metadata": {},
                }
            )
            if len(results) >= limit:
                break
        return results


def verify_model():
    semSearch = SemanticSearch()
    print(f"Model loaded: {semSearch.model}")
    print(f"Max sequence length: {semSearch.model.max_seq_length}")


def embed_text(text):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    sem_search = SemanticSearch()
    documents = get_movies()
    embeddings = sem_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def search_documents(query, limit):
    sem_search = SemanticSearch()
    documents = get_movies()
    sem_search.load_or_create_embeddings(documents)
    results = sem_search.search(query, limit)
    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results):
        print(f"{i + 1}. {result["title"]} (score: {result["score"]})")
        print(f"   {result['description'][:100]}...")
        print()


def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    groups = []
    for i in range(0, len(words), chunk_size):
        inner_i = i
        is_first_item = inner_i == 0

        if not is_first_item:
            inner_i = inner_i - overlap

        groups.append(" ".join(words[inner_i : inner_i + chunk_size]))

    print(f"Chunking {len(text)} characters")
    for i in range(len(groups)):
        print(f"{i + 1}. {groups[i]}")


def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    parsed_text = text.strip()

    if not parsed_text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", parsed_text)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]

    chunks = []

    i = 0

    while i < len(sentences):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            cleaned_sentences.append(chunk_sentence.strip())
        if not cleaned_sentences:
            continue

        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)

        i += max_chunk_size - overlap

    return chunks


def semantic_chunk_text(text: str, max_chunk_size: int, overlap: int):
    groups = semantic_chunk(text, max_chunk_size, overlap)

    print(f"Semantically chunking {len(text)} characters")
    for i in range(len(groups)):
        print(f"{i + 1}. {groups[i]}")


def embed_chunks():
    documents = get_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int):
    documents = get_movies()
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    results = chunked_semantic_search.search_chunks(query, limit)

    for i in range(len(results)):
        result = results[i]
        title = result["title"]
        score = result["score"]
        description = result["description"]
        print(f"\n{i + 1}. {title} (score: {score:.4f})")
        print(f"   {description}...")
