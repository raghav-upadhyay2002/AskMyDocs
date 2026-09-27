import html
import os
import time

import gradio as gr
from groq import APIConnectionError, APIStatusError, AuthenticationError, NotFoundError, RateLimitError

from src.embedder import get_model as load_embedder
from src.llm import CITATION_RE, GROUNDING_WARNING, MODEL, cited_numbers
from src.pipeline import ingest, query
from src.prompts import DEFAULT_PROMPT
from src.reranker import get_model as load_reranker

SAMPLE_PDF = "data/sample.pdf"
HAS_KEY = bool(os.getenv("GROQ_API_KEY"))  # from .env locally, or a Space secret when deployed

STYLES = [
    ("Strict — every sentence cited", "strict"),
    ("Detailed", "default"),
    ("Concise", "concise"),
]

EXAMPLES = [
    "What is this document about?",
    "Summarize the key findings.",
    "What data was collected, and how?",
    "What limitations are mentioned?",
]


# ---------------------------------------------------------------- rendering

def _status(state, title, detail=""):
    return (
        f'<div class="doc-status {state}"><span class="dot"></span><div>'
        f'<div class="doc-title">{html.escape(title)}</div>'
        f'<div class="doc-detail">{html.escape(detail)}</div>'
        f"</div></div>"
    )


EMPTY_STATUS = _status("empty", "No document yet", "Upload a PDF or try the sample paper.")
EMPTY_PLACEHOLDER = (
    "**Upload a PDF to get started.**<br>"
    "Answers cite passages like <span class='cite'>1</span> — expand *Sources* under each answer to read them."
)


def _ready_placeholder(name):
    return f"**{html.escape(name)} is ready.**<br>Ask anything about it, or start with a suggestion below."


def _render_answer(answer):
    body = answer.replace(GROUNDING_WARNING, "").strip()
    body = CITATION_RE.sub(lambda m: f'<span class="cite">{m.group(1)}</span>', body)
    if GROUNDING_WARNING in answer:
        body += "\n\n> ⚠️ **No citations found** — this answer may not be grounded in the document."
    return body


def _render_sources(sources, cited):
    parts = []
    for i, source in enumerate(sources, start=1):
        snippet = " ".join(source["text"].split())
        if len(snippet) > 320:
            snippet = snippet[:320].rsplit(" ", 1)[0] + " …"
        note = "" if i in cited else " <em>· not cited</em>"
        parts.append(
            f'<div class="source"><div class="source-head"><span class="cite">{i}</span> Page {source["page"]}{note}</div>'
            f'<div class="source-text">{html.escape(snippet)}</div></div>'
        )
    return "\n".join(parts)


def _assistant(content, **metadata):
    message = {"role": "assistant", "content": content}
    if metadata:
        message["metadata"] = metadata
    return message


def _error_message(e):
    if isinstance(e, RateLimitError):
        return "⏳ This demo runs on a free Groq key and has hit its usage limit. Please try again in a minute."
    if isinstance(e, AuthenticationError):
        return "🔑 Groq rejected the app's API key — the `GROQ_API_KEY` setting needs updating."
    if isinstance(e, NotFoundError):
        return f"🧩 The model `{MODEL}` isn't available on Groq anymore. Set the `GROQ_MODEL` environment variable to a current model."
    if isinstance(e, APIConnectionError):
        return "🌐 Couldn't reach Groq. Check your connection and try again."
    if isinstance(e, APIStatusError):
        return f"⚠️ Groq returned an error ({e.status_code}). Please try again in a moment."
    return f"⚠️ Something went wrong: {e}"


# ---------------------------------------------------------------- handlers

def index_document(pdf_path, old_index):
    """Build a fresh index whenever the uploaded file changes. Also resets the chat."""
    if old_index is not None:
        old_index.close()
    empty_chat = gr.update(value=[], placeholder=EMPTY_PLACEHOLDER)
    if not pdf_path:
        yield None, EMPTY_STATUS, empty_chat
        return

    name = os.path.basename(pdf_path)
    yield None, _status("loading", name, "Reading and indexing…"), empty_chat
    try:
        index = ingest(pdf_path)
    except ValueError as e:
        yield None, _status("error", name, str(e)), empty_chat
        return
    except Exception:
        yield None, _status("error", name, "Couldn't read this file — is it a valid PDF?"), empty_chat
        return

    pages = len({c["page"] for c in index.chunks})
    detail = f"{pages} page{'s' * (pages != 1)} · {len(index.chunks)} chunks · ready"
    yield index, _status("ready", name, detail), gr.update(value=[], placeholder=_ready_placeholder(name))


def add_question(question, history):
    question = (question or "").strip()
    if not question:
        return gr.update(), history, ""
    return gr.update(value="", interactive=False), history + [{"role": "user", "content": question}], question


def add_example(history, evt: gr.SelectData):
    return add_question(evt.value["text"], history)


def respond(history, question, index, style):
    if not question:
        yield history
        return
    if index is None:
        yield history + [_assistant("📄 Upload a PDF (or click **Try the sample paper**) first, then ask away.")]
        return
    if not HAS_KEY:
        yield history + [_assistant("🔑 No Groq API key configured. Set `GROQ_API_KEY` in `.env` (local) or as a Space secret.")]
        return

    yield history + [_assistant("", title="Searching the document…", status="pending")]

    started = time.time()
    try:
        result = query(index, question, prompt_name=style)
    except Exception as e:
        yield history + [_assistant(_error_message(e))]
        return

    sources = result["sources"]
    pages = ", ".join(str(p) for p in sorted({s["page"] for s in sources}))
    yield history + [
        _assistant(_render_answer(result["answer"])),
        _assistant(
            _render_sources(sources, cited_numbers(result["answer"])),
            title=f"Sources · page{'s' * (len(sources) != 1)} {pages}",
            status="done",
            duration=round(time.time() - started, 1),
        ),
    ]


# ---------------------------------------------------------------- layout

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display: none !important; }

.amd-header { display: flex; align-items: center; gap: 14px; padding: 12px 2px 4px; }
.amd-logo {
  width: 44px; height: 44px; flex: none; border-radius: 12px;
  display: grid; place-items: center; color: #fff;
  background: linear-gradient(135deg, #7c3aed, #4f46e5);
  box-shadow: 0 6px 18px rgba(124, 58, 237, .35);
}
.amd-header h1 { margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em; }
.amd-header p { margin: 2px 0 0; color: var(--body-text-color-subdued); font-size: .95rem; }

.panel { gap: 14px !important; }
.step { margin-bottom: -6px; }
.step p {
  margin: 0; font-size: .72rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--body-text-color-subdued);
}

.doc-status {
  display: flex; gap: 10px; align-items: flex-start; padding: 10px 12px;
  border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg);
  background: var(--background-fill-secondary);
}
.doc-status .dot { width: 8px; height: 8px; margin-top: 7px; flex: none; border-radius: 50%; background: var(--body-text-color-subdued); }
.doc-status.ready .dot { background: #22c55e; }
.doc-status.error .dot { background: #ef4444; }
.doc-status.loading .dot { background: #f59e0b; animation: amd-pulse 1s ease-in-out infinite; }
.doc-title { font-weight: 600; word-break: break-word; }
.doc-detail { font-size: .85rem; color: var(--body-text-color-subdued); }
.doc-status.error .doc-detail { color: #ef4444; }
@keyframes amd-pulse { 50% { opacity: .3; } }

.cite {
  display: inline-block; min-width: 1.45em; padding: 0 .4em; margin: 0 1px;
  border-radius: 999px; text-align: center; vertical-align: .1em;
  font-size: .72em; font-weight: 700; line-height: 1.55;
  color: #6d28d9; background: rgba(124, 58, 237, .13);
}
.dark .cite { color: #c4b5fd; background: rgba(167, 139, 250, .18); }

.source { padding: 10px 0; border-top: 1px solid var(--border-color-primary); }
.source:first-child { padding-top: 2px; border-top: none; }
.source-head { margin-bottom: 6px; font-size: .85rem; font-weight: 600; }
.source-head em { font-weight: 400; color: var(--body-text-color-subdued); }
.source-text {
  padding-left: 10px; border-left: 2px solid var(--color-accent-soft);
  font-size: .85rem; line-height: 1.55; color: var(--body-text-color);
}

.amd-stack { font-size: .8rem; color: var(--body-text-color-subdued); line-height: 1.5; }
.amd-stack code { font-size: .78rem; }

button.example {
  border: 1px solid var(--border-color-primary) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--background-fill-secondary) !important;
  transition: border-color .15s, background .15s;
}
button.example:hover { border-color: var(--color-accent) !important; background: var(--color-accent-soft) !important; }

#ask-row { align-items: stretch; }
#ask-btn { min-width: 90px; }
"""

HEADER = """
<div class="amd-header">
  <div class="amd-logo">
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h5"/>
    </svg>
  </div>
  <div>
    <h1>AskMyDocs</h1>
    <p>Ask questions about any PDF — every answer cites the passages it came from.</p>
  </div>
</div>
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

with gr.Blocks(title="AskMyDocs") as demo:
    index_state = gr.State(None, delete_callback=lambda index: index and index.close())
    pending_question = gr.State("")

    gr.HTML(HEADER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=300, elem_classes="panel"):
            gr.Markdown("1 · Document", elem_classes="step")
            pdf_input = gr.File(label="PDF", file_types=[".pdf"], show_label=False, height=130)
            sample_btn = gr.Button("Try the sample paper", size="sm", variant="secondary")
            doc_status = gr.HTML(EMPTY_STATUS)

            gr.Markdown("2 · Answer style", elem_classes="step")
            style = gr.Radio(STYLES, value=DEFAULT_PROMPT, show_label=False)

            gr.HTML(
                '<div class="amd-stack">Hybrid search (ChromaDB + BM25) → cross-encoder rerank → '
                f"<code>{html.escape(MODEL)}</code> on Groq, with a citation check on every answer.</div>"
            )

        with gr.Column(scale=8, min_width=360):
            clear_btn = gr.Button("Clear chat", size="sm", render=False)
            chatbot = gr.Chatbot(
                show_label=False,
                height=580,
                buttons=["copy", clear_btn],
                placeholder=EMPTY_PLACEHOLDER,
                examples=[{"text": q} for q in EXAMPLES],
            )
            with gr.Row(elem_id="ask-row"):
                question = gr.Textbox(
                    show_label=False,
                    placeholder="Ask a question about your document…",
                    scale=8,
                    max_lines=4,
                    autofocus=True,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1, elem_id="ask-btn")

    # Indexing — runs on upload, on "Try the sample paper", and resets when the file is removed.
    pdf_input.change(
        index_document,
        [pdf_input, index_state],
        [index_state, doc_status, chatbot],
        concurrency_limit=2,
    )
    sample_btn.click(lambda: SAMPLE_PDF, None, pdf_input)

    # Asking — echo the question immediately (and lock the box), then answer.
    asked = [
        gr.on(
            [question.submit, ask_btn.click],
            add_question,
            [question, chatbot],
            [question, chatbot, pending_question],
            queue=False,
        ),
        chatbot.example_select(add_example, chatbot, [question, chatbot, pending_question], queue=False),
    ]
    for event in asked:
        event.then(
            respond, [chatbot, pending_question, index_state, style], chatbot, concurrency_limit=8
        ).then(lambda: gr.update(interactive=True), None, question, queue=False)

    clear_btn.click(lambda: [], None, chatbot, queue=False)


if __name__ == "__main__":
    # Load both local models up front so the first question isn't slow.
    load_embedder()
    load_reranker()
    # Spaces turn on SSR by default, which skips Gradio's scoped copy of the custom CSS.
    demo.launch(theme=THEME, css=CSS, ssr_mode=False)
