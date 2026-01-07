import os
from sentence_transformers import SentenceTransformer
import numpy as np

from utils import get_movies

cache_movie_embeddings_path = "cache/movie_embeddings.npy"


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


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


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
