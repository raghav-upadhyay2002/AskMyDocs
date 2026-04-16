import fitz  # PyMuPDF


def load_pdf(file_path: str) -> str:
    """
    Opens a PDF and extracts all text from every page.
    Returns one big string with all the text.
    """
    doc = fitz.open(file_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[Page {page_num + 1}]\n{text}"

    doc.close()
    return full_text


if __name__ == "__main__":
    text = load_pdf("data/sample.pdf")
    print(text[:500])  
    print(f"\nTotal characters extracted: {len(text)}")