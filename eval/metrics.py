"""
Evaluation metrics for the RAG pipeline.

faithfulness_score  — is the answer grounded in retrieved chunks?
answer_relevance    — does the answer address the question?
citation_rate       — what % of answers include citations?
"""
import json
import os
import re
from src.llm import chat, cited_numbers

# Set JUDGE_MODEL to grade different answer models with the same judge. Unset, the judge
# uses the app's model and falls back like the app does when it's rate-limited.
JUDGE_MODEL = os.getenv("JUDGE_MODEL")


def citation_rate(answers):
    """Fraction of answers that contain at least one citation like [1]."""
    cited = sum(1 for a in answers if cited_numbers(a))
    return cited / len(answers) if answers else 0.0


def _llm_judge(question, answer, context):
    prompt = f"""\
You are evaluating a RAG system. Score the answer on two criteria.
Return ONLY a JSON object like: {{"faithfulness": 0.9, "relevance": 0.8}}

Faithfulness (0-1): Is every claim in the answer supported by the context? 1 = fully supported, 0 = made up.
Relevance (0-1): Does the answer actually address the question? 1 = fully addresses it, 0 = irrelevant.

Question: {question}
Context: {context}
Answer: {answer}
"""
    raw = chat(prompt, temperature=0.0, model=JUDGE_MODEL, max_wait=60)
    # Tolerate code fences or stray text around the JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0) if match else raw)


def evaluate_sample(question, answer, context_chunks):
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    try:
        scores = _llm_judge(question, answer, context)
        return {
            "faithfulness": float(scores.get("faithfulness", 0)),
            "relevance": float(scores.get("relevance", 0)),
        }
    except Exception as e:
        print(f"    [judge error] {type(e).__name__}: {e}")
        return {"faithfulness": 0.0, "relevance": 0.0}
