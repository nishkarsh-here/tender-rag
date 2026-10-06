"""Step 1: load the tender PDF and clean up the text.

In  : a PDF file in data/tenders/
Out : a list of LangChain Documents, one per page, with tidy text

Why this step exists: a tender is a PDF, and an LLM can only read text.
We use PyPDFLoader because it gives us one Document per page, so every
chunk later on can still say which page it came from. That page number
is what lets the final answer cite its source.
"""

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

TENDER_FOLDER = Path(__file__).resolve().parent.parent / "data" / "tenders"


def clean_text(text):
    """Tidy up the text that came out of the PDF.

    PDF text extraction leaves a lot of mess: blank lines everywhere,
    runs of spaces used for layout, and dotted lines from form fields.
    None of that carries meaning, and all of it wastes room in the chunk.
    This is the same regex clean-up used in the class RAG notebook.
    """
    text = re.sub(r"\.{4,}", " ", text)   # dotted leader lines in forms
    text = re.sub(r"_{4,}", " ", text)    # underscore fill-in-the-blank lines

    # Reflow the text so chunks hold readable sentences.
    #
    # Some tender PDFs place every single word as its own positioned block,
    # so the extracted text comes out as one word per line. If we leave that
    # alone, a chunk is a column of words instead of a sentence and retrieval
    # gets much worse. We detect it by the average line length.
    lines = [line for line in text.split("\n") if line.strip()]
    average_line_length = sum(len(line) for line in lines) / max(len(lines), 1)

    if average_line_length < 25:
        # Word-per-line PDF: every whitespace run becomes a single space.
        text = re.sub(r"\s+", " ", text)
    else:
        # Normal PDF: a single newline is just line wrapping, so it becomes a
        # space, but a blank line is a real paragraph break and is kept.
        text = re.sub(r"\n[ \t]*\n+", "\x00", text)
        text = text.replace("\n", " ")
        text = text.replace("\x00", "\n\n")
        text = re.sub(r"[ \t]{2,}", " ", text)

    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def load_tender_pdf(pdf_path):
    """Read one tender PDF into a list of Documents, one per page."""
    pdf_path = Path(pdf_path)
    pages = PyPDFLoader(str(pdf_path)).load()

    for page in pages:
        page.page_content = clean_text(page.page_content)
        # Keep only the metadata we actually use, so the citation is readable.
        # PyPDFLoader pages are 0-indexed; humans count from 1.
        page.metadata = {
            "tender": pdf_path.stem,
            "page": page.metadata.get("page", 0) + 1,
        }

    # A page that is blank or nearly blank (a divider, a stamp) adds nothing
    # to retrieval, so we drop it instead of indexing noise.
    return [p for p in pages if len(p.page_content) > 50]


def load_all_tenders(folder=TENDER_FOLDER):
    """Read every tender PDF in the folder."""
    documents = []
    for pdf_path in sorted(Path(folder).glob("*.pdf")):
        pages = load_tender_pdf(pdf_path)
        print(f"  loaded {pdf_path.name}: {len(pages)} pages with text")
        documents.extend(pages)
    return documents


if __name__ == "__main__":
    docs = load_all_tenders()
    print(f"\nTotal pages loaded: {len(docs)}")
    print(f"Total characters  : {sum(len(d.page_content) for d in docs)}")
    print("\n--- first 400 characters of the first page ---")
    print(docs[0].page_content[:400])
