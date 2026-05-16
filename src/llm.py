import os
from groq import Groq
from dotenv import load_dotenv
from src.prompts import PROMPTS, DEFAULT_PROMPT

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _build_context(chunks):
    return "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks))


def _is_grounded(answer, chunks):
    """Hallucination check: answer must cite at least one chunk."""
    return any(f"[{i+1}]" in answer for i in range(len(chunks)))


def ask(question, context_chunks, prompt_name=DEFAULT_PROMPT):
    context = _build_context(context_chunks)
    template = PROMPTS[prompt_name]
    prompt = template.format(context=context, question=question)

    response = get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature reduces hallucination
    )
    answer = response.choices[0].message.content

    if not _is_grounded(answer, context_chunks):
        answer += "\n\n⚠️ Warning: No citations found — answer may not be grounded in the document."

    return answer
