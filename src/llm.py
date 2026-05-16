import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = None

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client

def ask(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
    response = get_client().chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
