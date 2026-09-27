import uuid

import chromadb
from rank_bm25 import BM25Okapi

_client = chromadb.EphemeralClient()


def _tokenize(text):
    return text.lower().split()


def _normalize(scores):
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


class HybridIndex:
    """Hybrid search over one document: ChromaDB vectors (60%) + BM25 keywords (40%).

    Each index owns its own collection, so several documents (e.g. one per web
    session) can be indexed at once without overwriting each other.
    """

    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])
        self._collection = _client.create_collection(f"doc-{uuid.uuid4().hex}")
        self._collection.add(
            documents=[c["text"] for c in chunks],
            embeddings=embeddings,
            ids=[str(i) for i in range(len(chunks))],
        )

    def search(self, query_embedding, query_text, n_results=10):
        n_results = min(n_results, len(self.chunks))

        # Vector search
        vector_results = self._collection.query(query_embeddings=[query_embedding], n_results=n_results)
        vector_ids = [int(i) for i in vector_results["ids"][0]]
        vector_distances = vector_results["distances"][0]
        vector_scores_raw = _normalize([-d for d in vector_distances])

        # BM25 search over ALL chunks — take top n_results
        bm25_scores_all = self._bm25.get_scores(_tokenize(query_text))
        bm25_top_ids = sorted(range(len(bm25_scores_all)), key=lambda i: bm25_scores_all[i], reverse=True)[:n_results]
        bm25_scores_norm = _normalize(list(bm25_scores_all))

        # Union of vector and BM25 candidate IDs
        candidate_ids = {*vector_ids, *bm25_top_ids}

        vector_score_map = dict(zip(vector_ids, vector_scores_raw))
        combined = {
            idx: 0.6 * vector_score_map.get(idx, 0.0) + 0.4 * bm25_scores_norm[idx]
            for idx in candidate_ids
        }

        ranked = sorted(combined, key=combined.get, reverse=True)
        return [self.chunks[idx] for idx in ranked[:n_results]]

    def close(self):
        """Free the underlying Chroma collection."""
        try:
            _client.delete_collection(self._collection.name)
        except Exception:
            pass
