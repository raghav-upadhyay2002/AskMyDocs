import os
import re
import time

from dotenv import load_dotenv
from groq import Groq, RateLimitError

from src.prompts import PROMPTS, DEFAULT_PROMPT

load_dotenv()

# Groq retires models from time to time — override with GROQ_MODEL instead of editing code.
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Each Groq model has its own free-tier quota, so when one is rate-limited the next is tried
# right away instead of making the user wait. Comma-separated; set to "" to disable.
FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GROQ_FALLBACK_MODELS", "qwen/qwen3.8-27b,openai/gpt-oss-120b").split(",")
    if m.strip() and m.strip() != MODEL
]

# Caps answer length (and so how much of the per-minute token quota one question can use).
MAX_OUTPUT_TOKENS = 2048

GROUNDING_WARNING = "⚠️ Warning: No citations found — answer may not be grounded in the document."

# Matches [1], [2] and grouped forms like [1, 3].
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# gpt-oss sometimes falls back to its native 【1】 / 【1†source】 citation style.
_NATIVE_CITATION_RE = re.compile(r"【\s*(\d+(?:\s*,\s*\d+)*)[^】]*】")

# Replies the prompts tell the model to give when the context has no answer.
_DECLINE_PREFIXES = ("not found", "i don't have enough information")


_client = None


def get_client():
    global _client
    if _client is None:
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is not set — add it to .env (or as a Space secret).")
        # Retries are handled in chat(); the SDK's own would silently wait on every 429.
        _client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=0)
    return _client


def _complete(model, prompt, temperature):
    # gpt-oss models reason before answering; "low" keeps latency close to a non-reasoning model.
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    response = get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        **extra,
    )
    return (response.choices[0].message.content or "").strip()


def _retry_after(error):
    try:
        return float(error.response.headers.get("retry-after", 5))
    except (TypeError, ValueError):
        return 5.0


def chat(prompt, temperature=0.1, model=None, max_wait=20):
    """Single-turn completion. Returns the reply text.

    On a rate limit, moves straight on to the next of FALLBACK_MODELS (unless a specific
    `model` is requested). Only when every model is limited does it wait for the soonest
    reset — and it gives up rather than wait more than `max_wait` seconds in total.
    """
    models = [model] if model else [MODEL, *FALLBACK_MODELS]
    waited = 0.0
    while True:
        errors = []
        for m in models:
            try:
                return _complete(m, prompt, temperature)
            except RateLimitError as e:
                errors.append(e)
        wait = min(_retry_after(e) for e in errors)
        if waited + wait > max_wait:
            raise errors[0]
        time.sleep(wait)
        waited += wait


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


def ask(question, context_chunks, prompt_name=DEFAULT_PROMPT):
    context = _build_context(context_chunks)
    template = PROMPTS[prompt_name]
    prompt = template.format(context=context, question=question)

    answer = chat(prompt, temperature=0.1)
    answer = _NATIVE_CITATION_RE.sub(r"[\1]", answer)

    if not _is_grounded(answer, context_chunks) and not _is_decline(answer):
        answer += f"\n\n{GROUNDING_WARNING}"

    return answer
