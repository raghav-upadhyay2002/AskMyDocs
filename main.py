from src.pipeline import ingest, query


# STEP 1: Put a PDF in the data/ folder

PDF_PATH = "data/sample.pdf"


# STEP 2: Run ingestion (only need to do this once per PDF)

ingest(PDF_PATH)


# STEP 3: Ask questions about the document

questions = [
    "What is the main topic of this document?",
    "Summarize the key points.",
]

for question in questions:
    print(f"Q: {question}")
    answer = query(question)
    print(f"A: {answer}")
    print("-" * 60)