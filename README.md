# AskMyDocs — RAG Q&A System

Ask questions about any PDF and get answers with context pulled directly from the document.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a free Groq API key at console.groq.com
#    Then add it to .env:
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Put a PDF in the data/ folder
# 4. Update the PDF_PATH in main.py
# 5. Run it
python main.py
```

## Project structure

```
askmydocs/
├── src/
│   ├── loader.py       # PDF → raw text
│   ├── chunker.py      # raw text → chunks
│   ├── embedder.py     # chunks → vectors
│   ├── vectorstore.py  # store & search vectors (ChromaDB)
│   ├── llm.py          # query Groq LLaMA with context
│   └── pipeline.py     # orchestrates ingestion + query
├── data/               # put your PDFs here
├── main.py             # entry point
├── requirements.txt
└── .env                # your API key (never commit this)
```

## How it works

**Ingestion (run once per document):**
PDF → extract text → split into 500-char chunks → embed each chunk → store in ChromaDB

**Query (run every time you ask a question):**
Question → embed question → find top-3 similar chunks → send chunks + question to LLaMA → get answer