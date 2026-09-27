def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


def chunk_pages(pages, chunk_size=500, overlap=50):
    """Chunk each page separately so every chunk maps to a single (1-based) page number."""
    chunks = []
    for page_num, text in enumerate(pages, start=1):
        for piece in chunk_text(text, chunk_size, overlap):
            if piece.strip():
                chunks.append({"text": piece, "page": page_num})
    return chunks
