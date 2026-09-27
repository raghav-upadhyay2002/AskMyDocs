---
title: AskMyDocs
emoji: 📄
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# AskMyDocs — RAG Q&A System

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/raghavupadhyay/askmydocs)

Ask questions about any PDF and get answers with citations pulled directly from the document — each one linked to the page it came from.

> **Try it live →** https://huggingface.co/spaces/raghavupadhyay/askmydocs

## Features

- **Hybrid search** — combines vector similarity (60%) and BM25 keyword matching (40%) for better retrieval
- **Reranking** — cross-encoder model re-scores top candidates before sending to the LLM
- **Citation enforcement** — every answer references the exact chunks it was derived from, with page numbers
- **Hallucination detection** — answers without citations are flagged automatically
- **Prompt versioning** — swap between `default`, `strict`, and `concise` prompt styles
- **Web app** — chat UI with expandable sources; each browser session gets its own document index
- **Swappable LLM** — any Groq chat model via the `GROQ_MODEL` env var (default `openai/gpt-oss-20b`)
- **Rate-limit fallback** — if that model hits its free-tier limit, the next one in `GROQ_FALLBACK_MODELS` answers instead (each Groq model has its own quota)
- **Whole-document mode** — short documents (up to ~12K characters, e.g. a resume) are sent to the LLM in full, so broad questions like "summarize this" cover everything
- **Evaluation system** — automated quality checks with faithfulness, relevance, and citation rate metrics
- **CI pipeline** — GitHub Actions fails the build if quality drops below thresholds

## Project structure

```
askmydocs/
├── src/
│   ├── loader.py        # PDF → text per page
│   ├── chunker.py       # pages → overlapping chunks tagged with page numbers
│   ├── embedder.py      # chunks → vectors (all-MiniLM-L6-v2)
│   ├── vectorstore.py   # HybridIndex: ChromaDB + BM25, one per document
│   ├── reranker.py      # cross-encoder reranking
│   ├── llm.py           # Groq LLM with citations + hallucination check
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
├── app.py               # Gradio web app (also runs the Hugging Face Space)
├── main.py              # command-line example
├── requirements.txt
└── .env                 # your API key (never commit this)
```

## Setup

```bash
# 1. Create a virtualenv and install dependencies (Python 3.10+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Get a free Groq API key at console.groq.com
echo "GROQ_API_KEY=your_key_here" > .env

# 3a. Run the web app → http://127.0.0.1:7860
python app.py

# 3b. …or the command-line example (edit PDF_PATH in main.py)
python main.py
```

Groq retires models from time to time. If answers start failing with a "model not found" error, pick a current model from [console.groq.com/docs/models](https://console.groq.com/docs/models) and set it — no code change needed:

```bash
echo "GROQ_MODEL=openai/gpt-oss-120b" >> .env
```

**Deploying to Hugging Face Spaces:** add `GROQ_API_KEY` as a secret under the Space's *Settings → Variables and secrets*. The app reads the key from there, so visitors never need their own — and it never appears in the code.

## How it works

**Ingestion (run once per document):**
```
PDF → extract text per page → split into 500-char chunks (50-char overlap)
    → embed each chunk → store in ChromaDB + BM25 index
```

**Query (run every time you ask a question):**
```
Question → embed → hybrid search (vector + BM25) → top 10 candidates
         → rerank with cross-encoder → top 3 chunks
         → send to the LLM with citation prompt → answer + sources (with page numbers)
```

Short documents (≤ ~12K characters) skip search and reranking: every chunk goes to the LLM.

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
