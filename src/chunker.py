import re

_WHITESPACE = re.compile(r"\s")


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into ~chunk_size-char chunks overlapping by up to ~overlap chars.

    Cuts land on whitespace so words are never split in half (unless a single
    "word" is longer than half a chunk, e.g. a long URL).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start + chunk_size // 2:
                end = cut
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Step back for the overlap, then forward to the next word boundary
        start = end - overlap
        boundary = _WHITESPACE.search(text, start, end)
        if boundary:
            start = boundary.end()
    return chunks


def chunk_pages(pages, chunk_size=500, overlap=50):
    """Chunk each page separately so every chunk maps to a single (1-based) page number."""
    chunks = []
    for page_num, text in enumerate(pages, start=1):
        for piece in chunk_text(text, chunk_size, overlap):
            if piece.strip():
                chunks.append({"text": piece.strip(), "page": page_num})
    return chunks
