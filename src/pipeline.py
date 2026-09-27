from src.loader import load_pages
from src.chunker import chunk_pages
from src.embedder import embed
from src.vectorstore import HybridIndex
from src.reranker import rerank
from src.llm import ask
from src.prompts import DEFAULT_PROMPT


def ingest(pdf_path):
    """PDF → page-tagged chunks → embeddings → hybrid index. Returns the index."""
    chunks = chunk_pages(load_pages(pdf_path))
    if not chunks:
        raise ValueError("No extractable text found in this PDF — it may be a scanned image.")
    index = HybridIndex(chunks, embed([c["text"] for c in chunks]))
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")
    return index


def retrieve(index, question, n_candidates=10, top_k=3):
    # Hybrid search: vector + BM25, returns top n_candidates
    candidates = index.search(embed([question])[0], question, n_results=n_candidates)

    # Rerank candidates, keep top_k
    return rerank(question, candidates, top_k=top_k)


def query(index, question, prompt_name=DEFAULT_PROMPT, api_key=None):
    """Answer a question about an ingested document.

    Returns {"answer": str, "sources": [{"text", "page"}, ...]}, where sources[i]
    is the chunk the answer cites as [i+1].
    """
    sources = retrieve(index, question)

    # Ask LLM with citations + hallucination check
    answer = ask(question, [c["text"] for c in sources], prompt_name=prompt_name, api_key=api_key)
    return {"answer": answer, "sources": sources}
