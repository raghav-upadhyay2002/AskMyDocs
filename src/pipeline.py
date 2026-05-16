from src.loader import import_file
from src.chunker import chunk_text
from src.embedder import embed
from src.vectorstore import add_chunks, search
from src.llm import ask

def ingest(pdf_path):
    text = import_file(pdf_path)
    chunks = chunk_text(text)
    embeddings = embed(chunks)
    add_chunks(chunks, embeddings)
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")

def query(question):
    question_embedding = embed([question])[0]
    context_chunks = search(question_embedding)
    return ask(question, context_chunks)
