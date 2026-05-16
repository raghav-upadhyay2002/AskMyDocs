import chromadb
from rank_bm25 import BM25Okapi

_client = chromadb.Client()
_collection = None
_bm25 = None
_all_chunks = []


def get_collection(name="docs"):
    global _collection
    if _collection is None:
        _collection = _client.get_or_create_collection(name)
    return _collection


def add_chunks(chunks, embeddings):
    global _bm25, _all_chunks
    _all_chunks = chunks
    tokenized = [chunk.lower().split() for chunk in chunks]
    _bm25 = BM25Okapi(tokenized)

    col = get_collection()
    col.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))],
    )


def _normalize(scores):
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def search(query_embedding, query_text, n_results=10):
    col = get_collection()

    # Vector search — retrieve more candidates for reranking
    vector_results = col.query(query_embeddings=[query_embedding], n_results=n_results)
    vector_docs = vector_results["documents"][0]
    vector_ids = [int(i) for i in vector_results["ids"][0]]
    vector_distances = vector_results["distances"][0]
    # ChromaDB returns distances (lower = better), convert to similarity
    vector_scores = _normalize([-d for d in vector_distances])

    # BM25 search over all chunks
    bm25_scores_all = _bm25.get_scores(query_text.lower().split())
    # Get BM25 scores only for the vector-retrieved chunks
    bm25_scores = _normalize([bm25_scores_all[i] for i in vector_ids])

    # Combine: 60% vector, 40% BM25
    combined = {}
    for idx, doc, vscore, bscore in zip(vector_ids, vector_docs, vector_scores, bm25_scores):
        combined[idx] = {"doc": doc, "score": 0.6 * vscore + 0.4 * bscore}

    ranked = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked]
