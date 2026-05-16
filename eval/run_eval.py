"""
Runs evaluation against eval/dataset.json and exits with code 1 if quality thresholds are not met.
Usage: python -m eval.run_eval
CI will fail the build if this script exits with code 1.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pipeline import ingest, query
from src.embedder import embed
from src.vectorstore import search
from src.reranker import rerank
from eval.metrics import evaluate_sample, citation_rate

DATASET_PATH = "eval/dataset.json"
PDF_PATH = "data/sample.pdf"
RESULTS_PATH = "eval/results.json"

# Quality thresholds — build fails if any are not met
THRESHOLDS = {
    "faithfulness": 0.70,
    "relevance": 0.70,
    "citation_rate": 0.80,
}

# How many samples to evaluate (set to None for full dataset)
MAX_SAMPLES = 20


def run():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Run eval/generate_dataset.py first.")
        sys.exit(1)

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    samples = dataset[:MAX_SAMPLES] if MAX_SAMPLES else dataset
    print(f"Evaluating {len(samples)} samples...")

    ingest(PDF_PATH)

    results = []
    answers = []

    for i, item in enumerate(samples):
        question = item["question"]
        answer = query(question)
        answers.append(answer)

        # Get context chunks used for this answer (re-run retrieval)
        q_emb = embed([question])[0]
        candidates = search(q_emb, question, n_results=10)
        top_chunks = rerank(question, candidates, top_k=3)

        scores = evaluate_sample(question, answer, top_chunks)
        results.append({
            "question": question,
            "expected": item.get("answer", ""),
            "actual": answer,
            **scores,
        })
        print(f"  [{i+1}/{len(samples)}] faithfulness={scores['faithfulness']:.2f} relevance={scores['relevance']:.2f}")

    # Aggregate
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    cit_rate = citation_rate(answers)

    summary = {
        "samples_evaluated": len(results),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_relevance": round(avg_relevance, 3),
        "citation_rate": round(cit_rate, 3),
        "thresholds": THRESHOLDS,
        "passed": True,
    }

    print("\n--- Results ---")
    print(f"Faithfulness:  {avg_faithfulness:.3f}  (threshold: {THRESHOLDS['faithfulness']})")
    print(f"Relevance:     {avg_relevance:.3f}  (threshold: {THRESHOLDS['relevance']})")
    print(f"Citation rate: {cit_rate:.3f}  (threshold: {THRESHOLDS['citation_rate']})")

    failed = []
    if avg_faithfulness < THRESHOLDS["faithfulness"]:
        failed.append(f"faithfulness {avg_faithfulness:.3f} < {THRESHOLDS['faithfulness']}")
    if avg_relevance < THRESHOLDS["relevance"]:
        failed.append(f"relevance {avg_relevance:.3f} < {THRESHOLDS['relevance']}")
    if cit_rate < THRESHOLDS["citation_rate"]:
        failed.append(f"citation_rate {cit_rate:.3f} < {THRESHOLDS['citation_rate']}")

    summary["passed"] = len(failed) == 0
    summary["failures"] = failed

    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")

    if failed:
        print("\n❌ BUILD FAILED — quality thresholds not met:")
        for f in failed:
            print(f"   - {f}")
        sys.exit(1)
    else:
        print("\n✅ All quality thresholds passed.")
        sys.exit(0)


if __name__ == "__main__":
    run()
