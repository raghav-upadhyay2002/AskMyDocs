import os
import re
import tempfile
import gradio as gr

_ingested_id = None


def ingest_pdf(pdf_file, groq_key):
    global _ingested_id

    if not groq_key or not groq_key.strip():
        yield gr.update(value="⚠️ Please enter your Groq API key.", visible=True), gr.update(interactive=True)
        return
    if pdf_file is None:
        yield gr.update(value="⚠️ Please upload a PDF.", visible=True), gr.update(interactive=True)
        return

    os.environ["GROQ_API_KEY"] = groq_key.strip()

    file_id = f"{pdf_file.name}_{os.path.getsize(pdf_file.name)}"
    if _ingested_id == file_id:
        yield gr.update(value="✅ Already indexed — ready to answer questions.", visible=True), gr.update(interactive=True)
        return

    # Show spinner immediately, disable button
    yield gr.update(value="⏳ Indexing document, please wait…", visible=True), gr.update(interactive=False)

    # Reset vectorstore state
    import src.vectorstore as vs
    vs._collection = None
    vs._bm25 = None
    vs._all_chunks = []
    try:
        vs._client.delete_collection("docs")
    except Exception:
        pass

    from src.pipeline import ingest
    ingest(pdf_file.name)
    _ingested_id = file_id

    yield gr.update(value=f"✅ Indexed **{os.path.basename(pdf_file.name)}** — ask your questions below.", visible=True), gr.update(interactive=True)


def answer_question(question, groq_key, prompt_style):
    if not question or not question.strip():
        yield "", gr.update(interactive=True)
        return
    if not groq_key or not groq_key.strip():
        yield "⚠️ Add your Groq API key and index a PDF first.", gr.update(interactive=True)
        return
    if _ingested_id is None:
        yield "⚠️ Upload and index a PDF first.", gr.update(interactive=True)
        return

    yield "⏳ Thinking…", gr.update(interactive=False)

    os.environ["GROQ_API_KEY"] = groq_key.strip()

    from src.pipeline import query
    answer = query(question.strip(), prompt_name=prompt_style)
    answer = re.sub(r"\[(\d+)\]", r"**[\1]**", answer)

    yield answer, gr.update(interactive=True)


with gr.Blocks(
    title="AskMyDocs",
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.violet, secondary_hue=gr.themes.colors.blue),
    css="""
    .container { max-width: 800px; margin: auto; }
    .answer-box textarea { font-family: monospace; font-size: 0.9rem; line-height: 1.7; }
    """,
) as demo:

    gr.Markdown("""
# 📄 AskMyDocs
**RAG-powered PDF Q&A** — every answer is grounded in your document with `[1]` `[2]` citations.
""")

    with gr.Row():
        with gr.Column(scale=1):
            groq_key = gr.Textbox(
                label="Groq API Key",
                placeholder="gsk_...",
                type="password",
                info="Free key at console.groq.com",
            )
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            prompt_style = gr.Dropdown(
                choices=["strict", "default", "concise"],
                value="strict",
                label="Answer style",
            )
            ingest_btn = gr.Button("📥 Index Document", variant="primary")
            ingest_status = gr.Markdown("")

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question",
                placeholder="What is the main finding of this study?",
                lines=2,
            )
            ask_btn = gr.Button("Ask ➤", variant="primary")
            answer_output = gr.Markdown(label="Answer", elem_classes=["answer-box"])

            gr.Examples(
                examples=[
                    ["What is the title of this paper?"],
                    ["What is the purpose of this study?"],
                    ["Who are the authors?"],
                    ["What research method was used?"],
                    ["What are the key findings?"],
                ],
                inputs=question_input,
            )

    gr.Markdown("""
---
**Stack:** BM25 + Vector hybrid search · CrossEncoder reranker · LLaMA 3.1 8B via Groq · CI/CD eval (faithfulness 77%, citation rate 90%)
""")

    ingest_btn.click(fn=ingest_pdf, inputs=[pdf_input, groq_key], outputs=[ingest_status, ingest_btn])
    ask_btn.click(fn=answer_question, inputs=[question_input, groq_key, prompt_style], outputs=[answer_output, ask_btn])
    question_input.submit(fn=answer_question, inputs=[question_input, groq_key, prompt_style], outputs=[answer_output, ask_btn])

if __name__ == "__main__":
    demo.launch()
