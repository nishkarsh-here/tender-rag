"""Gradio app: ask a question about a tender and get a grounded answer.

Run with:  python app.py

This file is only the user interface. All the RAG work happens in the rag/
package, one file per step. Nothing here does any retrieval or any prompting
itself - it calls step 6, which calls steps 4 and 5.

The tender cards shown on the home screen come from data/tender_cards.json,
which build_tender_cards.py produces by running the same pipeline.
"""

import html
import json
from pathlib import Path

import gradio as gr

from rag.step3_embed_store import CHROMA_FOLDER, build_vector_store, load_vector_store
from rag.step4_retrieve import TOP_K
from rag.step6_generate import answer_with_fallback

ROOT = Path(__file__).resolve().parent
CARDS_FILE = ROOT / "data" / "tender_cards.json"
ALL_TENDERS = "All tenders"


def ensure_index():
    """Build the vector store if it is not there yet.

    A fresh clone has no chroma_db/ folder, because it is not committed. Rather
    than failing with a confusing empty result, we build it on first start.
    It takes about a minute for three tenders, and only happens once.
    """
    try:
        if load_vector_store().get()["ids"]:
            return
    except Exception:
        pass

    print("No vector store found. Building it now (about a minute) ...")
    from rag.step1_load import load_all_tenders
    from rag.step2_chunk import split_into_chunks

    chunks = split_into_chunks(load_all_tenders())
    build_vector_store(chunks)
    print(f"Indexed {len(chunks)} chunks into {CHROMA_FOLDER}")


ensure_index()

EXAMPLE_QUESTIONS = [
    "What is the earnest money deposit?",
    "What is the estimated cost put to bid?",
    "What is the last date and time for closing of bids?",
    "What happens to the EMD if a bidder withdraws the bid?",
    "Who should the demand draft be drawn in favour of?",
    "Is there any exemption from paying the EMD?",
]

CSS = """
.tender-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.tender-card {
  border: 1px solid var(--border-color-primary);
  border-left: 3px solid #b8860b;
  border-radius: 6px;
  padding: 14px 16px;
  background: var(--background-fill-secondary);
}
.tender-card .tc-title { font-weight: 600; font-size: 14px; line-height: 1.35; margin-bottom: 3px; }
.tender-card .tc-org { font-size: 12px; opacity: .72; margin-bottom: 10px; line-height: 1.35; }
.tender-card dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 0; font-size: 12.5px; }
.tender-card dt { opacity: .62; white-space: nowrap; }
.tender-card dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; word-break: break-word; }
.tender-card .tc-foot { margin-top: 10px; font-size: 11.5px; opacity: .55; font-family: ui-monospace, monospace; }
.proof {
  border: 1px solid var(--border-color-primary);
  border-radius: 6px;
  padding: 11px 13px;
  margin-bottom: 9px;
  background: var(--background-fill-secondary);
  font-size: 12.5px;
  line-height: 1.6;
}
.proof .p-src {
  font-family: ui-monospace, monospace;
  font-size: 11px;
  letter-spacing: .04em;
  text-transform: uppercase;
  opacity: .62;
  display: block;
  margin-bottom: 5px;
}
.statusline { font-family: ui-monospace, monospace; font-size: 12px; opacity: .7; }
footer { display: none !important; }
"""


def load_cards():
    if CARDS_FILE.exists():
        return json.loads(CARDS_FILE.read_text())
    return []


CARDS = load_cards()


def card_html():
    """The home screen: what tenders are loaded, and what each one is."""
    if not CARDS:
        return (
            "<p>No tender cards yet. Run "
            "<code>python build_tender_cards.py</code> to create them.</p>"
        )

    blocks = []
    for card in CARDS:
        rows = "".join(
            f"<dt>{label}</dt><dd>{html.escape(str(card.get(key, '')))}</dd>"
            for label, key in [
                ("Reference", "reference"),
                ("Closing", "closing"),
                ("EMD", "emd"),
                ("Value", "value"),
            ]
        )
        blocks.append(
            f"""<div class="tender-card">
  <div class="tc-title">{html.escape(card['title'])}</div>
  <div class="tc-org">{html.escape(card['organisation'])}</div>
  <dl>{rows}</dl>
  <div class="tc-foot">{html.escape(card['file'])} · {card['pages']} pages</div>
</div>"""
        )
    return f'<div class="tender-grid">{"".join(blocks)}</div>'


def dropdown_choices():
    """Readable labels, so the selector is not a list of file names."""
    choices = [(ALL_TENDERS, ALL_TENDERS)]
    for card in CARDS:
        title = card["title"]
        if len(title) > 44:
            title = title[:44].rstrip() + "..."
        short_org = card["organisation"].split("(")[0].split("-")[0].strip()
        if len(short_org) > 34:
            short_org = short_org[:34].rstrip() + "..."
        choices.append((f"{title} - {short_org}", card["tender"]))
    return choices


def ask(question, tender_choice):
    """Called when the user presses Ask. Returns answer, proof and a status line."""
    if not question.strip():
        return "Please type a question.", "", ""

    tender = None if tender_choice == ALL_TENDERS else tender_choice
    answer, chunks, model_used = answer_with_fallback(question, tender=tender)

    # Show the retrieved clauses the answer was built from, so the user can
    # check it against the real document rather than taking it on trust.
    proofs = []
    for chunk in chunks:
        source = f"{chunk.metadata['tender']} · page {chunk.metadata['page']}"
        text = html.escape(chunk.page_content.strip())
        proofs.append(
            f'<div class="proof"><span class="p-src">{html.escape(source)}'
            f"</span>{text}</div>"
        )

    status = (
        f'<div class="statusline">retrieved {len(chunks)} chunks '
        f"(k={TOP_K}) &nbsp;·&nbsp; answered by {html.escape(model_used)}</div>"
    )
    return answer, "".join(proofs), status


with gr.Blocks(title="Tender RAG", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("## Tender RAG")
    gr.Markdown(
        "Ask a question about a government tender. The answer comes only from "
        "the tender PDFs below, and every answer shows the clauses it was "
        "built from. If the tenders do not cover the question, the app says so "
        "instead of guessing."
    )

    gr.Markdown("### Tenders loaded")
    gr.HTML(card_html())

    gr.Markdown("### Ask a question")
    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What is the earnest money deposit?",
            lines=2,
            scale=3,
        )
        tender_dropdown = gr.Dropdown(
            label="Search in",
            choices=dropdown_choices(),
            value=ALL_TENDERS,
            scale=2,
        )

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_box, label="Try one")

    ask_button = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=4)
    status_html = gr.HTML()

    with gr.Accordion("Show the clauses this answer came from", open=False):
        proof_html = gr.HTML()

    ask_button.click(
        fn=ask,
        inputs=[question_box, tender_dropdown],
        outputs=[answer_box, proof_html, status_html],
    )
    question_box.submit(
        fn=ask,
        inputs=[question_box, tender_dropdown],
        outputs=[answer_box, proof_html, status_html],
    )

if __name__ == "__main__":
    demo.launch()
