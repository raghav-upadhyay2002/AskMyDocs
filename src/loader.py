import fitz

def import_file(file_path):
    with fitz.open(file_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

if __name__ == "__main__":
    test = import_file('data/sample.pdf')
    print(test)