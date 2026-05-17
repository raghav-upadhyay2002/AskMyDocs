"""
Evaluation metrics for the RAG pipeline.

faithfulness_score  — is the answer grounded in retrieved chunks?
answer_relevance    — does the answer address the question?
citation_rate       — what % of answers include citations?
"""
import re
import time
from groq import RateLimitError
from src.llm import get_client


def citation_rate(answers):
    """Fraction of answers that contain at least one citation like [1]."""
    cited = sum(1 for a in answers if re.search(r"\[\d+\]", a))
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
    for attempt in range(5):
        try:
            response = get_client().chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            break
        except RateLimitError:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    import json
    return json.loads(raw.strip())


def evaluate_sample(question, answer, context_chunks):
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    try:
        scores = _llm_judge(question, answer, context)
        return {
            "faithfulness": float(scores.get("faithfulness", 0)),
            "relevance": float(scores.get("relevance", 0)),
        }
    except Exception:
        return {"faithfulness": 0.0, "relevance": 0.0}
