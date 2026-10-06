"""Gradio app: ask a question about a tender and get a grounded answer.

Run with:  python app.py

This file is only the user interface. All the RAG work happens in the rag/
package, one file per step. Nothing here does any retrieval or any prompting
itself - it calls step 6, which calls steps 4 and 5.

The tender cards shown on the home screen come from data/tender_cards.json,
which build_tender_cards.py produces by running the same pipeline.
"""

import argparse
import html
import json
from pathlib import Path

import gradio as gr

from rag.step2_chunk import CHUNK_OVERLAP, CHUNK_SIZE
from rag.step3_embed_store import (
    CHROMA_FOLDER,
    EMBEDDING_MODEL,
    build_vector_store,
    load_vector_store,
)
from rag.step4_retrieve import TOP_K
from rag.step5_prompt import TEMPLATE
from rag.step6_generate import (
    CHAT_MODEL,
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    TEMPERATURE,
    answer_with_fallback,
)

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
.tour-step {
  display: grid; grid-template-columns: 30px 1fr; gap: 14px;
  padding: 15px 0; border-top: 1px solid var(--border-color-primary);
}
.tour-step:first-child { border-top: 0; }
.tour-n {
  width: 24px; height: 24px; border-radius: 50%;
  background: #b8860b; color: #fff;
  display: grid; place-items: center;
  font-size: 12px; font-family: ui-monospace, monospace; margin-top: 2px;
}
.tour-body { min-width: 0; }
.tour-file { font-family: ui-monospace, monospace; font-size: 13px; font-weight: 600; }
.tour-what { font-size: 13.5px; opacity: .85; margin: 4px 0 8px; line-height: 1.55; }
.tour-tech { display: flex; flex-wrap: wrap; gap: 6px; }
.tour-tech span {
  font-family: ui-monospace, monospace; font-size: 11px;
  border: 1px solid var(--border-color-primary); border-radius: 10px;
  padding: 2px 9px; opacity: .85;
}
.tour-live { font-size: 11.5px; opacity: .6; margin-top: 7px; font-family: ui-monospace, monospace; }
.tour-note {
  border-left: 3px solid #b8860b; padding: 10px 14px; margin: 14px 0;
  background: var(--background-fill-secondary); border-radius: 0 5px 5px 0;
  font-size: 13px; line-height: 1.6;
}
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


def tour_html():
    """The guided tour: what each stage does, and what it is built with.

    The numbers are read from the code and the live vector store, so this page
    cannot drift out of date with what the system actually does.
    """
    try:
        indexed = len(load_vector_store().get()["ids"])
    except Exception:
        indexed = "not built yet"

    pages = sum(c["pages"] for c in CARDS) if CARDS else "?"

    steps = [
        (
            "rag/step1_load.py",
            "Read each tender PDF into one Document per page, then clean the text. "
            "Keeping pages separate is what lets an answer cite a page number later. "
            "Two of our tenders put every word on its own line, so the cleaner detects "
            "that and reflows them into sentences.",
            ["LangChain", "PyPDFLoader", "pypdf", "re (regex)"],
            f"{len(CARDS)} tenders, {pages} pages with text",
        ),
        (
            "rag/step2_chunk.py",
            "Cut the pages into small overlapping chunks. A tender is far longer than "
            "the model's context window, and a smaller chunk retrieves more precisely "
            "because its embedding is about one thing.",
            ["LangChain", "RecursiveCharacterTextSplitter"],
            f"chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} -> {indexed} chunks",
        ),
        (
            "rag/step3_embed_store.py",
            "Turn every chunk into a vector that represents its meaning, and store it. "
            "This is what lets a question about the \"EMD\" find a clause that says "
            "\"earnest money deposit\" instead of failing on the wording.",
            ["sentence-transformers", "HuggingFaceEmbeddings", "Chroma"],
            f"{EMBEDDING_MODEL.split('/')[-1]}, 384 dimensions, saved to chroma_db/",
        ),
        (
            "rag/step4_retrieve.py",
            "Embed the question and return the chunks closest to it by cosine "
            "similarity. Picking a tender adds a metadata filter, so the search is "
            "still by meaning but only inside that document.",
            ["Chroma", "as_retriever", "cosine similarity", "metadata filter"],
            f"top k={TOP_K} of {indexed} chunks",
        ),
        (
            "rag/step5_prompt.py",
            "Paste the retrieved clauses into the prompt as context, and tell the "
            "model to answer only from them and to say so when they do not cover the "
            "question. This is the \"Augmented\" in Retrieval Augmented Generation.",
            ["LangChain", "ChatPromptTemplate"],
            "{context} + {question}, with a refuse-if-absent instruction",
        ),
        (
            "rag/step6_generate.py",
            "Send the finished prompt to the model and return the answer. The main "
            "path is the LCEL chain from class. A second path goes through LiteLLM, "
            "which retries on a different model if the first call fails.",
            ["LangChain", "init_chat_model", "LCEL", "StrOutputParser", "Groq", "LiteLLM"],
            f"{CHAT_MODEL}, temperature {TEMPERATURE}",
        ),
    ]

    blocks = []
    for i, (file, what, tech, live) in enumerate(steps, start=1):
        chips = "".join(f"<span>{html.escape(t)}</span>" for t in tech)
        blocks.append(
            f"""<div class="tour-step">
  <div class="tour-n">{i}</div>
  <div class="tour-body">
    <div class="tour-file">{html.escape(file)}</div>
    <div class="tour-what">{what}</div>
    <div class="tour-tech">{chips}</div>
    <div class="tour-live">{html.escape(str(live))}</div>
  </div>
</div>"""
        )
    return "".join(blocks)


def fallback_html():
    return f"""<div class="tour-note">
<b>If the model call fails</b><br>
<code>{html.escape(PRIMARY_MODEL)}</code> &nbsp;-&gt;&nbsp; call raises &nbsp;-&gt;&nbsp;
<code>{html.escape(FALLBACK_MODEL)}</code><br>
One try/except in <code>rag/step6_generate.py</code>. The same prompt goes to the
second model, and the status line under every answer names the model that
actually replied, so the switch is visible rather than silent.
</div>"""


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


with gr.Blocks(title="Tender RAG") as demo:
    gr.Markdown("## Tender RAG")
    gr.Markdown(
        "Ask a question about a government tender. The answer comes only from "
        "the tender PDFs below, and every answer shows the clauses it was "
        "built from. If the tenders do not cover the question, the app says so "
        "instead of guessing."
    )

    with gr.Tab("Ask"):
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

    with gr.Tab("How it works"):
        gr.Markdown("### The pipeline, stage by stage")
        gr.Markdown(
            "Each stage is one file in `rag/`, named after the stage. The chips "
            "under each one are the libraries that stage uses, and the grey line "
            "is what it is doing right now in this running app."
        )
        gr.HTML(tour_html())

        gr.Markdown("### Falling back to a second model")
        gr.HTML(fallback_html())

        gr.Markdown("### The prompt we send")
        gr.Markdown(
            "Everything above exists to fill in `{context}` below. The retrieved "
            "clauses go in there, and the rules are what stop the model answering "
            "from its own general knowledge."
        )
        gr.Code(value=TEMPLATE, language=None, label="rag/step5_prompt.py")

        gr.Markdown("### What we measured")
        gr.Markdown(
            "**Chunk size.** We started at 1000 characters, the class default, and "
            "it worked badly: page 1 of a tender is a summary table, so at that size "
            "it becomes one chunk whose embedding averages many topics. Measured on "
            "six questions we checked by hand in the PDFs - 300: 5/6, 500: 3/6, "
            "800: 3/6, 1000: 1/6, 1500: 2/6. We use 300.\n\n"
            "**How many chunks.** On twelve hand-checked questions - k=8: 10/12, "
            "k=10: 11/12, k=20: 12/12. We use 10, because k=20 doubles the prompt to "
            "chase one tender reference number.\n\n"
            "**Hybrid retrieval.** We built the BM25 + semantic ensemble from class "
            "and measured it. It fixed one question and broke two others, and simply "
            "raising k did better. We kept semantic search. The numbers and the "
            "reasoning are in `rag/step4_retrieve.py` under `WHY_NOT_HYBRID`."
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Tender RAG app.")
    parser.add_argument(
        "--share",
        action="store_true",
        help="also create a temporary public link (useful when demonstrating)",
    )
    args = parser.parse_args()

    # Gradio 6 takes the theme and the stylesheet here rather than on Blocks.
    demo.launch(theme=gr.themes.Soft(), css=CSS, share=args.share)
