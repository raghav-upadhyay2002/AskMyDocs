from sentence_transformers import CrossEncoder

_model = None

def get_model():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model

def rerank(question, chunks, top_k=3):
    pairs = [(question, chunk) for chunk in chunks]
    scores = get_model().predict(pairs)
    ranked = sorted(zip(scores, chunks), reverse=True)
    return [chunk for _, chunk in ranked[:top_k]]
