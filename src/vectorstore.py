import chromadb

_client = chromadb.Client()
_collection = None

def get_collection(name="docs"):
    global _collection
    if _collection is None:
        _collection = _client.get_or_create_collection(name)
    return _collection

def add_chunks(chunks, embeddings):
    col = get_collection()
    col.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))],
    )

def search(query_embedding, n_results=3):
    col = get_collection()
    results = col.query(query_embeddings=[query_embedding], n_results=n_results)
    return results["documents"][0]
