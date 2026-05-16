"""
Run this once to generate eval/dataset.json from your PDF.
Usage: python -m eval.generate_dataset
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.loader import import_file
from src.chunker import chunk_text
from src.llm import get_client

PDF_PATH = "data/sample.pdf"
OUTPUT_PATH = "eval/dataset.json"
QUESTIONS_PER_CHUNK = 2
MAX_CHUNKS = 60  # generates up to 120 Q&A pairs; raise for 200


def generate_qa_for_chunk(chunk, chunk_id):
    prompt = f"""\
You are building an evaluation dataset for a RAG system.
Given the text chunk below, generate {QUESTIONS_PER_CHUNK} question-answer pairs.
The questions must be answerable ONLY from this chunk.
Return a JSON array like:
[{{"question": "...", "answer": "...", "chunk_id": {chunk_id}}}]
Return ONLY the JSON array, no explanation.

Chunk:
{chunk}
"""
    response = get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def main():
    print(f"Loading {PDF_PATH}...")
    text = import_file(PDF_PATH)
    chunks = chunk_text(text)
    chunks = chunks[:MAX_CHUNKS]
    print(f"Generating Q&A from {len(chunks)} chunks...")

    dataset = []
    for i, chunk in enumerate(chunks):
        try:
            pairs = generate_qa_for_chunk(chunk, i)
            dataset.extend(pairs)
            print(f"  Chunk {i+1}/{len(chunks)}: {len(pairs)} pairs generated")
        except Exception as e:
            print(f"  Chunk {i+1} failed: {e}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to {OUTPUT_PATH} ({len(dataset)} Q&A pairs)")


if __name__ == "__main__":
    main()
