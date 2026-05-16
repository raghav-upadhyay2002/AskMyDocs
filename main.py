from src.pipeline import ingest, query

PDF_PATH = "data/sample.pdf"

ingest(PDF_PATH)

questions = [
    "What is the main topic of this document?",
    "Summarize the key points.",
]

# Try different prompt versions: "default", "strict", "concise"
PROMPT = "default"

for question in questions:
    print(f"Q: {question}")
    answer = query(question, prompt_name=PROMPT)
    print(f"A: {answer}")
    print("-" * 60)
