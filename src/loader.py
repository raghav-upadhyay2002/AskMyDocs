import pymupdf


def load_pages(file_path):
    """Return the extracted text of each page, in page order."""
    with pymupdf.open(file_path) as doc:
        return [page.get_text() for page in doc]


if __name__ == "__main__":
    pages = load_pages("data/sample.pdf")
    print(f"{len(pages)} pages")
    print(pages[0])
