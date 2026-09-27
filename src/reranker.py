from sentence_transformers import CrossEncoder

_model = None

def get_model():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

def rerank(question, chunks, top_k=3):
    """Re-score chunks ({"text", "page"} dicts) with the cross-encoder and keep the best top_k."""
    if not chunks:
        return []
    scores = get_model().predict([(question, chunk["text"]) for chunk in chunks])
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
