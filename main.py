from src.pipeline import ingest, query

PDF_PATH = "data/sample.pdf"

index = ingest(PDF_PATH)

questions = [
    "What is the main topic of this document?",
    "Summarize the key points.",
]

# Try different prompt versions: "default", "strict", "concise"
PROMPT = "default"

for question in questions:
    print(f"Q: {question}")
    result = query(index, question, prompt_name=PROMPT)
    print(f"A: {result['answer']}")
    for i, source in enumerate(result["sources"], start=1):
        print(f"   [{i}] page {source['page']}")
    print("-" * 60)
