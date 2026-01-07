from lib.search_utils import GOLD_DATASET, MOVIES_JSON_F
from lib.hybrid_search import HybridSearch
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit

    with open(GOLD_DATASET, "r") as f:
        test_cases = json.load(f)["test_cases"]
    
    with open(MOVIES_JSON_F, "r") as f:
        documents = json.load(f)["movies"]
    search = HybridSearch(documents)
    
    metrics = {}
    for test_case in test_cases:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        
        results = search.rrf_search(query, 60, limit)
        if results is not None:
            retrieved_titles = [result["doc"]["title"] for result in results]
            precision_score = sum(1 for title in retrieved_titles if title in relevant_docs) / limit
            recall_score = sum(1 for title in retrieved_titles if title in relevant_docs) / len(relevant_docs)
            f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
        else:
            raise ValueError("Failed to get results")
        
        metrics[query] = {
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "retrieved_docs": retrieved_titles,
            "relevant_docs": relevant_docs
        }
    
    print(f"k={limit}\n")
    for query in metrics.keys():
        entry = metrics[query]
        precision = entry["precision_score"]
        recall = entry["recall_score"]
        f1 = entry["f1_score"]
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(entry['retrieved_docs'])}")
        print(f"  - Relevant: {', '.join(entry['relevant_docs'])}\n")


if __name__ == "__main__":
    main()