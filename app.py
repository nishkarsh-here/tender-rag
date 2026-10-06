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
from langchain_community.document_loaders import PyPDFLoader

from rag.step1_load import clean_text, load_tender_pdf
from rag.step2_chunk import CHUNK_OVERLAP, CHUNK_SIZE, split_into_chunks
from rag.step3_embed_store import (
    CHROMA_FOLDER,
    EMBEDDING_MODEL,
    build_vector_store,
    load_vector_store,
)
from rag.step4_retrieve import TOP_K, retrieve_chunks
from rag.step5_prompt import TEMPLATE, build_prompt, format_context
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
.tour-panel {
  border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: 18px 20px; background: var(--background-fill-secondary);
}
.tour-head { margin-bottom: 10px; }
.tour-stage {
  font-family: ui-monospace, monospace; font-size: 11px; letter-spacing: .08em;
  text-transform: uppercase; color: #b8860b;
}
.tour-title { font-size: 18px; font-weight: 600; margin: 3px 0 2px; }
.tour-panel .tour-file { font-family: ui-monospace, monospace; font-size: 12.5px; opacity: .7; font-weight: 400; }
.tour-panel p { font-size: 13.5px; line-height: 1.6; margin: 12px 0; }
.tour-panel pre {
  background: var(--background-fill-primary); border: 1px solid var(--border-color-primary);
  border-radius: 5px; padding: 10px 12px; font-size: 11.5px; line-height: 1.55;
  white-space: pre-wrap; word-break: break-word; margin: 8px 0; overflow-x: auto;
}
.tour-prompt { max-height: 420px; overflow-y: auto; }
.tour-ba { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.tour-ba b { font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; opacity: .7; }
.tour-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }
.tour-tbl td { padding: 7px 8px; border-bottom: 1px solid var(--border-color-primary); }
.tour-tbl td.num { font-family: ui-monospace, monospace; font-weight: 600; width: 62px; }
.tour-answer {
  border-left: 3px solid #b8860b; padding: 12px 14px; margin: 12px 0;
  background: var(--background-fill-primary); border-radius: 0 5px 5px 0;
  font-size: 14px; line-height: 1.6;
}
@media (max-width: 620px) { .tour-ba { grid-template-columns: 1fr; } }
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


# ---------------------------------------------------------------------------
# The guided tour: run one question through the pipeline, one stage at a time.
# ---------------------------------------------------------------------------

TOUR_STEPS = [
    ("1. Load the PDF", "rag/step1_load.py", ["LangChain", "PyPDFLoader", "pypdf", "regex"]),
    ("2. Split into chunks", "rag/step2_chunk.py", ["LangChain", "RecursiveCharacterTextSplitter"]),
    ("3. Embed and store", "rag/step3_embed_store.py", ["sentence-transformers", "HuggingFaceEmbeddings", "Chroma"]),
    ("4. Retrieve", "rag/step4_retrieve.py", ["Chroma", "as_retriever", "cosine similarity"]),
    ("5. Build the prompt", "rag/step5_prompt.py", ["LangChain", "ChatPromptTemplate"]),
    ("6. Generate the answer", "rag/step6_generate.py", ["init_chat_model", "LCEL", "Groq", "LiteLLM"]),
]


def _panel(body):
    return f'<div class="tour-panel">{body}</div>'


def _chips(tech):
    return '<div class="tour-tech">' + "".join(f"<span>{html.escape(t)}</span>" for t in tech) + "</div>"


def run_tour_stage(step, question, tender):
    """Produce the display for one stage, actually running that stage."""
    title, file, tech = TOUR_STEPS[step]
    head = (
        f'<div class="tour-head"><div><span class="tour-stage">Stage {step + 1} of 6</span>'
        f'<div class="tour-title">{html.escape(title)}</div>'
        f'<div class="tour-file">{html.escape(file)}</div></div></div>' + _chips(tech)
    )

    tender_name = None if tender == ALL_TENDERS else tender
    sample = tender_name or (CARDS[0]["tender"] if CARDS else None)

    if step == 0:
        pages = load_tender_pdf(ROOT / "data" / "tenders" / f"{sample}.pdf")
        raw = PyPDFLoader(str(ROOT / "data" / "tenders" / f"{sample}.pdf")).load()[0].page_content
        body = (
            "<p>The PDF becomes one Document per page. Keeping pages apart is what "
            "lets the final answer cite a page number.</p>"
            "<p>Some tenders place every word as its own block, so the raw text comes "
            "out one word per line. <code>clean_text()</code> spots that and reflows it.</p>"
            f"<div class='tour-live'>{html.escape(sample)}.pdf gave {len(pages)} pages with text</div>"
            "<div class='tour-ba'><div><b>Raw from the PDF</b>"
            f"<pre>{html.escape(repr(raw[:150]))}</pre></div>"
            "<div><b>After clean_text()</b>"
            f"<pre>{html.escape(clean_text(raw)[:260])}</pre></div></div>"
        )

    elif step == 1:
        pages = load_tender_pdf(ROOT / "data" / "tenders" / f"{sample}.pdf")
        chunks = split_into_chunks(pages)
        shown = "".join(
            f"<pre>page {c.metadata['page']} · {html.escape(c.page_content[:190])}</pre>"
            for c in chunks[4:7]
        )
        body = (
            "<p>A tender is far longer than the model's context window, and a smaller "
            "chunk retrieves more precisely because its embedding is about one thing.</p>"
            f"<div class='tour-live'>chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP} "
            f"→ {len(pages)} pages became {len(chunks)} chunks</div>"
            "<p><b>Three chunks in a row</b> — notice the overlap at their edges:</p>" + shown
        )

    elif step == 2:
        from rag.step3_embed_store import get_embedding_model
        emb = get_embedding_model()
        vec = emb.embed_query(question or "earnest money deposit")
        try:
            total = len(load_vector_store().get()["ids"])
        except Exception:
            total = "?"
        import numpy as np

        def cos(a, b):
            va, vb = emb.embed_query(a), emb.embed_query(b)
            return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

        pairs = [
            ("last date for submission of bids", "Date and time of closing of bids is 11-05-2026"),
            ("earnest money deposit", "EMD amount payable by the bidder"),
            ("last date for submission of bids", "The glove box must have an oxygen sensor."),
        ]
        rows = "".join(
            f"<tr><td class='num'>{cos(a, b):+.3f}</td><td>{html.escape(a)}</td>"
            f"<td>{html.escape(b)}</td></tr>" for a, b in pairs
        )
        body = (
            "<p>Each chunk becomes a list of numbers standing for its meaning. Chunks "
            "that mean similar things get vectors pointing in similar directions, so "
            "meaning can be compared arithmetically.</p>"
            f"<div class='tour-live'>{EMBEDDING_MODEL.split('/')[-1]} · "
            f"{len(vec)} numbers per chunk · {total} chunks stored in chroma_db/</div>"
            f"<pre>your question → [{', '.join(f'{v:.3f}' for v in vec[:6])}, ...]</pre>"
            "<p><b>Cosine similarity, computed live.</b> Different words, same meaning "
            "scores high. Note the abbreviation in the middle row scoring poorly — a "
            "small model does not reliably link \"EMD\" to its full form.</p>"
            f"<table class='tour-tbl'>{rows}</table>"
        )

    elif step == 3:
        chunks = retrieve_chunks(question, tender=tender_name)
        rows = "".join(
            f"<div class='proof'><span class='p-src'>{html.escape(c.metadata['tender'])} · "
            f"page {c.metadata['page']}</span>{html.escape(c.page_content[:230])}</div>"
            for c in chunks[:4]
        )
        body = (
            "<p>The question is embedded with the same model, and Chroma returns the "
            "chunks whose vectors are closest. Only these reach the prompt.</p>"
            f"<div class='tour-live'>k={TOP_K} · searched "
            f"{html.escape(tender_name or 'all three tenders')} · showing the top 4</div>"
            + rows
        )

    elif step == 4:
        chunks = retrieve_chunks(question, tender=tender_name)
        body = (
            "<p>This is the <b>Augmented</b> in Retrieval Augmented Generation. The "
            "retrieved clauses are pasted in where <code>{context}</code> sits, and the "
            "rules stop the model answering from its own general knowledge.</p>"
            "<p><b>This is the exact text sent to the model:</b></p>"
            f"<pre class='tour-prompt'>{html.escape(build_prompt(question, chunks)[:2600])}\n...</pre>"
        )

    else:
        answer, chunks, model_used = answer_with_fallback(question, tender=tender_name)
        srcs = []
        for c in chunks:
            tag = f"{c.metadata['tender']}, page {c.metadata['page']}"
            if tag not in srcs:
                srcs.append(tag)
        body = (
            "<p>The prompt goes to the model and the answer comes back. The main path "
            "is the LCEL chain from class. The LiteLLM path used here retries on a "
            "second model if the first call fails.</p>"
            f"<div class='tour-answer'>{html.escape(answer)}</div>"
            f"<div class='tour-live'>answered by {html.escape(model_used)} · "
            f"temperature {TEMPERATURE} · sources: {html.escape(', '.join(srcs[:4]))}</div>"
        )

    return _panel(head + body), f"Stage {step + 1} of 6"


def tour_start(question, tender):
    q = question.strip() or "What is the earnest money deposit?"
    panel, label = run_tour_stage(0, q, tender)
    return 0, q, panel, label


def tour_move(step, question, tender, delta):
    step = max(0, min(len(TOUR_STEPS) - 1, step + delta))
    panel, label = run_tour_stage(step, question, tender)
    return step, panel, label


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

    with gr.Tab("Guided tour"):
        gr.Markdown("### Watch a question move through the pipeline")
        gr.Markdown(
            "Type a question, press **Start the tour**, then step through the six "
            "stages. Each stage actually runs, so what you see is this question "
            "going through this system, not a picture of one."
        )
        with gr.Row():
            tour_q = gr.Textbox(
                label="Question for the tour",
                value="What is the earnest money deposit?",
                lines=1, scale=3,
            )
            tour_tender = gr.Dropdown(
                label="Search in", choices=dropdown_choices(),
                value=ALL_TENDERS, scale=2,
            )
        with gr.Row():
            tour_start_btn = gr.Button("Start the tour", variant="primary")
            tour_back_btn = gr.Button("< Back")
            tour_next_btn = gr.Button("Next >")

        tour_label = gr.Markdown("")
        tour_panel = gr.HTML()
        tour_step = gr.State(0)

        tour_start_btn.click(
            fn=tour_start, inputs=[tour_q, tour_tender],
            outputs=[tour_step, tour_q, tour_panel, tour_label],
        )
        tour_next_btn.click(
            fn=lambda st, q, t: tour_move(st, q, t, 1),
            inputs=[tour_step, tour_q, tour_tender],
            outputs=[tour_step, tour_panel, tour_label],
        )
        tour_back_btn.click(
            fn=lambda st, q, t: tour_move(st, q, t, -1),
            inputs=[tour_step, tour_q, tour_tender],
            outputs=[tour_step, tour_panel, tour_label],
        )

    with gr.Tab("Tech stack"):
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
