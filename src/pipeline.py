from src.loader import import_file
from src.chunker import chunk_text
from src.embedder import embed
from src.vectorstore import add_chunks, search
from src.reranker import rerank
from src.llm import ask
from src.prompts import DEFAULT_PROMPT


def ingest(pdf_path):
    text = import_file(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed(chunks)
    add_chunks(chunks, embeddings)
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")


def query(question, prompt_name=DEFAULT_PROMPT):
    # Embed question
    question_embedding = embed([question])[0]

    # Hybrid search: vector + BM25, returns top 10 candidates
    candidates = search(question_embedding, question, n_results=10)

    # Rerank candidates, keep top 3
    top_chunks = rerank(question, candidates, top_k=3)

    # Ask LLM with citations + hallucination check
    return ask(question, top_chunks, prompt_name=prompt_name)
