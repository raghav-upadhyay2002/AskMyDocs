# AskMyDocs — RAG Q&A System

Ask questions about any PDF and get answers with citations pulled directly from the document.

## Features

- **Hybrid search** — combines vector similarity (60%) and BM25 keyword matching (40%) for better retrieval
- **Reranking** — cross-encoder model re-scores top candidates before sending to the LLM
- **Citation enforcement** — every answer references the exact chunks it was derived from
- **Hallucination detection** — answers without citations are flagged automatically
- **Prompt versioning** — swap between `default`, `strict`, and `concise` prompt styles
- **Evaluation system** — automated quality checks with faithfulness, relevance, and citation rate metrics
- **CI pipeline** — GitHub Actions fails the build if quality drops below thresholds

## Project structure

```
askmydocs/
├── src/
│   ├── loader.py        # PDF → raw text
│   ├── chunker.py       # raw text → overlapping chunks
│   ├── embedder.py      # chunks → vectors (all-MiniLM-L6-v2)
│   ├── vectorstore.py   # hybrid search: ChromaDB + BM25
│   ├── reranker.py      # cross-encoder reranking
│   ├── llm.py           # Groq LLaMA with citations + hallucination check
│   ├── prompts.py       # versioned prompt templates
│   └── pipeline.py      # orchestrates ingestion + query
├── eval/
│   ├── generate_dataset.py  # generate Q&A pairs from PDF
│   ├── metrics.py           # faithfulness, relevance, citation rate
│   └── run_eval.py          # evaluation runner (exits 1 if thresholds not met)
├── .github/
│   └── workflows/
│       └── eval.yml     # CI pipeline
├── data/                # put your PDFs here
├── main.py              # entry point
├── requirements.txt
└── .env                 # your API key (never commit this)
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Groq API key at console.groq.com
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Put a PDF in the data/ folder and update PDF_PATH in main.py

# 4. Run
python main.py
```

## How it works

**Ingestion (run once per document):**
```
PDF → extract text → split into 500-char chunks (50-char overlap)
    → embed each chunk → store in ChromaDB + BM25 index
```

**Query (run every time you ask a question):**
```
Question → embed → hybrid search (vector + BM25) → top 10 candidates
         → rerank with cross-encoder → top 3 chunks
         → send to LLaMA with citation prompt → answer
```

## Prompt versions

Change `PROMPT` in `main.py` to switch styles:

| Prompt | Behaviour |
|---|---|
| `default` | Detailed answer with citations |
| `strict` | Every sentence must cite a chunk |
| `concise` | 1-2 sentence answer with citations |

## Evaluation

**Step 1 — Generate a dataset from your PDF (run once):**
```bash
python -m eval.generate_dataset
```
Generates ~120 Q&A pairs and saves them to `eval/dataset.json`. Commit this file.

**Step 2 — Run evaluation locally:**
```bash
python -m eval.run_eval
```
Scores 20 samples for faithfulness, relevance, and citation rate. Exits with code 1 if any threshold is not met.

**Quality thresholds (build fails if not met):**

| Metric | Threshold |
|---|---|
| Faithfulness | ≥ 0.70 |
| Relevance | ≥ 0.70 |
| Citation rate | ≥ 0.80 |

## CI pipeline

Every push to `main` and every pull request automatically runs the evaluation on GitHub Actions.

To set it up, add your Groq API key to GitHub:
> Repo → Settings → Secrets and variables → Actions → New repository secret
> - Name: `GROQ_API_KEY`
> - Value: your key
