import os
import re
import time
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from src.prompts import PROMPTS, DEFAULT_PROMPT

load_dotenv()

# Groq retires models from time to time — override with GROQ_MODEL instead of editing code.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

GROUNDING_WARNING = "⚠️ Warning: No citations found — answer may not be grounded in the document."

# Matches [1], [2] and grouped forms like [1, 3].
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# gpt-oss sometimes falls back to its native 【1】 / 【1†source】 citation style.
_NATIVE_CITATION_RE = re.compile(r"【\s*(\d+(?:\s*,\s*\d+)*)[^】]*】")

# Replies the prompts tell the model to give when the context has no answer.
_DECLINE_PREFIXES = ("not found", "i don't have enough information")


@lru_cache(maxsize=64)
def _client_for(api_key):
    return Groq(api_key=api_key)


def get_client(api_key=None):
    """Groq client for the given key, falling back to GROQ_API_KEY from the environment."""
    key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
    if not key:
        raise ValueError("No Groq API key provided. Set GROQ_API_KEY or pass api_key.")
    return _client_for(key)


def chat(prompt, temperature=0.1, api_key=None):
    """Single-turn completion with retry/backoff on rate limits. Returns the reply text."""
    # gpt-oss models reason before answering; "low" keeps latency close to a non-reasoning model.
    extra = {"reasoning_effort": "low"} if MODEL.startswith("openai/gpt-oss") else {}
    for attempt in range(5):
        try:
            response = get_client(api_key).chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **extra,
            )
            return (response.choices[0].message.content or "").strip()
        except RateLimitError:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def cited_numbers(answer):
    """Set of chunk numbers cited in the answer, e.g. {1, 3}."""
    return {int(n) for group in CITATION_RE.findall(answer) for n in group.split(",")}


def _build_context(chunks):
    return "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks))


def _is_grounded(answer, chunks):
    """Hallucination check: answer must cite at least one chunk."""
    return any(1 <= n <= len(chunks) for n in cited_numbers(answer))


def _is_decline(answer):
    return answer.lower().replace("’", "'").startswith(_DECLINE_PREFIXES)


def ask(question, context_chunks, prompt_name=DEFAULT_PROMPT, api_key=None):
    context = _build_context(context_chunks)
    template = PROMPTS[prompt_name]
    prompt = template.format(context=context, question=question)

    answer = chat(prompt, temperature=0.1, api_key=api_key)
    answer = _NATIVE_CITATION_RE.sub(r"[\1]", answer)

    if not _is_grounded(answer, context_chunks) and not _is_decline(answer):
        answer += f"\n\n{GROUNDING_WARNING}"

    return answer
