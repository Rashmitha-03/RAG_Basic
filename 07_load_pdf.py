import pymupdf

PDF_PATH = "data/mod5.pdf"   # <- change this


def load_pdf(path):
    """Return a list of (page_number, text) tuples."""
    doc = pymupdf.open(path)
    return [(i + 1, page.get_text()) for i, page in enumerate(doc)]


pages = load_pdf(PDF_PATH)
print("Total pages:", len(pages))
for page_no, text in pages[:3]:
    clean = " ".join(text.split())
    print(f"\nPage {page_no} ({len(clean)} characters):")
    print(clean[:200])