import argparse
import json

from hybrid_search import rrf_search


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open("data/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)["test_cases"]
        for dataset in golden_dataset:
            query = dataset["query"]

            print(f"- Query: {query}")

            relevant_docs = dataset["relevant_docs"]
            results = rrf_search(query, 60, limit)

            relevant_retrieved = 0
            total_relevant = len(relevant_docs)
            total_retrieved = len(results)
            result_values = results.values()
            result_titles = []

            for result_value in result_values:
                result_titles.append(result_value["title"])

            for relavant_doc in relevant_docs:

                if relavant_doc in result_titles:
                    relevant_retrieved += 1

            recall = 0
            if total_relevant > 0:
                recall = relevant_retrieved / total_relevant

            precision = relevant_retrieved / total_retrieved
            f1 = 2 * (precision * recall) / (precision + recall)

            print(f"    - Precision@{limit}: {precision:.4f}")
            print(f"    - Recall@{limit}: {recall:.4f}")
            print(f"    - F1 Score: {f1:.4f}")
            print(f"    - Retrieved: {", ".join(result_titles)}")
            print(f"    - Relevant: {", ".join(relevant_docs)}")
            print("")


if __name__ == "__main__":
    main()
