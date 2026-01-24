from hybrid_search import rrf_search
from lib.llm_helpers import (
    citation_response,
    question_response,
    rag_response,
    summarize_response,
)


def rag_search(query: str):
    results = rrf_search(query)
    rag = rag_response(query, results)

    print("Search Results:")
    for id in results:
        result = results[id]
        print(f"    - {result["title"]}")

    print("")
    print("RAG Response:")
    print(rag)


def summarize_search(query: str, limit: int):
    results = rrf_search(query, 60, limit)
    rag = summarize_response(query, results)

    print("Search Results:")
    for id in results:
        result = results[id]
        print(f"    - {result["title"]}")

    print("")
    print("LLM Response:")
    print(rag)


def citations_search(query: str, limit: int):
    results = rrf_search(query, 60, limit)
    rag = citation_response(query, results)

    print("Search Results:")
    for id in results:
        result = results[id]
        print(f"    - {result["title"]}")

    print("")
    print("LLM Response:")
    print(rag)


def question_search(query: str, limit: int):
    results = rrf_search(query, 60, limit)
    rag = question_response(query, results)

    print("Search Results:")
    for id in results:
        result = results[id]
        print(f"    - {result["title"]}")

    print("")
    print("LLM Response:")
    print(rag)
