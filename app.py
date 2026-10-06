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
import os
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
from tour import fallback_html, tour_html, tour_move, tour_start

from check_eligibility import (
    ReportUnavailable,
    check,
    company_by_id,
    format_company,
    load_companies,
)
from rag.step6_generate import (
    CHAT_MODEL,
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    TEMPERATURE,
    answer_with_fallback,
    compare_with_and_without_rag,
    generate_answer,
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
.cmp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.cmp-col {
  border: 1px solid var(--border-color-primary); border-radius: 8px;
  padding: 16px 18px; background: var(--background-fill-secondary); min-width: 0;
}
.cmp-col.bad { border-top: 3px solid #b4432f; }
.cmp-col.good { border-top: 3px solid #1f7a4d; }
.cmp-label {
  font-family: ui-monospace, monospace; font-size: 11px; letter-spacing: .08em;
  text-transform: uppercase; opacity: .72; margin-bottom: 4px;
}
.cmp-sub { font-size: 12px; opacity: .6; margin-bottom: 11px; line-height: 1.45; }
.cmp-body { font-size: 13.5px; line-height: 1.62; white-space: pre-wrap; word-break: break-word; }
.cmp-src { font-family: ui-monospace, monospace; font-size: 11px; opacity: .6; margin-top: 11px; }
@media (max-width: 620px) { .cmp-grid { grid-template-columns: 1fr; } }
.verdict {
  border-radius: 8px; padding: 15px 18px; margin-bottom: 14px;
  border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
}
.verdict.yes { border-left: 4px solid #1f7a4d; }
.verdict.no { border-left: 4px solid #b4432f; }
.verdict.maybe { border-left: 4px solid #b8860b; }
.verdict-head { font-size: 17px; font-weight: 600; margin-bottom: 6px; }
.verdict-sum { font-size: 13.5px; line-height: 1.6; opacity: .9; }
.crit { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 6px; }
.crit th {
  text-align: left; font-size: 11px; letter-spacing: .07em; text-transform: uppercase;
  opacity: .6; padding: 0 10px 7px 0; border-bottom: 1px solid var(--border-color-primary);
}
.crit td { padding: 9px 10px 9px 0; border-bottom: 1px solid var(--border-color-primary); vertical-align: top; }
.crit td.v { white-space: nowrap; width: 86px; }
.pill {
  font-family: ui-monospace, monospace; font-size: 10.5px; padding: 2px 8px;
  border-radius: 10px; border: 1px solid currentColor;
}
.pill.met { color: #1f7a4d; }
.pill.notmet { color: #b4432f; }
.pill.unclear { color: #b8860b; }
.crit td.req { font-weight: 600; width: 30%; }
.crit td.why { opacity: .82; line-height: 1.5; }
.crit td.pg { font-family: ui-monospace, monospace; font-size: 11px; opacity: .55; white-space: nowrap; width: 62px; }
.missing { margin-top: 14px; font-size: 13px; }
.missing li { margin-bottom: 4px; }
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
DEFAULT_TENDER = CARDS[0]["tender"] if CARDS else None


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


def compare_rag(question, tender_choice):
    """Answer the same question with and without the retrieved context."""
    question = question.strip()
    if not question:
        return "<p>Type a question first.</p>"

    tender = None if tender_choice == ALL_TENDERS else tender_choice
    without, with_rag, chunks = compare_with_and_without_rag(question, tender=tender)

    srcs = []
    for c in chunks:
        tag = f"{c.metadata['tender']}, page {c.metadata['page']}"
        if tag not in srcs:
            srcs.append(tag)

    return f"""<div class="cmp-grid">
  <div class="cmp-col bad">
    <div class="cmp-label">Without RAG</div>
    <div class="cmp-sub">The same model, the same question, no tender text.
      It has never read this document.</div>
    <div class="cmp-body">{html.escape(without.strip()[:1100])}</div>
    <div class="cmp-src">no sources - nothing was retrieved</div>
  </div>
  <div class="cmp-col good">
    <div class="cmp-label">With RAG</div>
    <div class="cmp-sub">The same model, the same question, with the retrieved
      clauses pasted into the prompt.</div>
    <div class="cmp-body">{html.escape(with_rag.strip()[:1100])}</div>
    <div class="cmp-src">{html.escape(' · '.join(srcs[:4]))}</div>
  </div>
</div>"""



def company_choices():
    return [(c["name"], c["id"]) for c in load_companies()]


def company_profile_text(company_id):
    company = company_by_id(company_id)
    return format_company(company) if company else ""


def _pill(verdict):
    key = {"Met": "met", "Not met": "notmet"}.get(verdict, "unclear")
    return f'<span class="pill {key}">{html.escape(verdict)}</span>'


def run_eligibility(company_id, profile_text, tender_choice):
    """Check one company against one tender, using the RAG pipeline."""
    tender = None if tender_choice == ALL_TENDERS else tender_choice
    if not tender:
        return "<p>Pick one tender. Eligibility is per tender, so \"All tenders\" does not apply.</p>"

    base = company_by_id(company_id) or {"name": "Company", "profile": {}}
    # Whatever is in the box is what gets checked, so an edited or uploaded
    # profile is used as typed rather than the saved one.
    company = {"name": base["name"], "profile": {"Profile": profile_text}}

    try:
        report, chunks = check(company, tender)
    except ReportUnavailable as error:
        return f"<div class='verdict maybe'><div class='verdict-head'>Could not produce a report</div><div class='verdict-sum'>{html.escape(str(error))}</div></div>"

    cls = {"Likely eligible": "yes", "Likely not eligible": "no"}.get(report.overall, "maybe")
    rows = "".join(
        f"<tr><td class='v'>{_pill(c.verdict)}</td>"
        f"<td class='req'>{html.escape(c.requirement)}</td>"
        f"<td class='why'>{html.escape(c.reason)}</td>"
        f"<td class='pg'>{html.escape(str(c.tender_page).replace('page page ', 'p').replace('page ', 'p'))}</td></tr>"
        for c in report.criteria
    )
    missing = ""
    if report.missing_documents:
        items = "".join(f"<li>{html.escape(d)}</li>" for d in report.missing_documents)
        missing = f"<div class='missing'><b>Documents the profile does not mention</b><ul>{items}</ul></div>"

    pages = sorted({c.metadata["page"] for c in chunks})
    return f"""<div class="verdict {cls}">
  <div class="verdict-head">{html.escape(report.overall)}</div>
  <div class="verdict-sum">{html.escape(report.summary)}</div>
</div>
<table class="crit">
  <tr><th>Verdict</th><th>Requirement from the tender</th><th>Why</th><th>Page</th></tr>
  {rows}
</table>
{missing}
<div class="statusline">read {len(chunks)} chunks from pages {html.escape(str(pages))} of {html.escape(tender)}</div>"""


def upload_tender(file_obj):
    """Add an uploaded tender PDF to the index so it can be queried."""
    if file_obj is None:
        return "No file chosen.", gr.update(), gr.update()

    src = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    if src.suffix.lower() != ".pdf":
        return "That is not a PDF.", gr.update(), gr.update()

    dest = ROOT / "data" / "tenders" / src.name
    dest.write_bytes(src.read_bytes())

    try:
        pages = load_tender_pdf(dest)
        if not pages:
            dest.unlink(missing_ok=True)
            return ("No text could be read from that PDF. It is probably a scan, "
                    "and we did not add OCR.", gr.update(), gr.update())
        chunks = split_into_chunks(pages)
        load_vector_store().add_documents(chunks)
    except Exception as error:
        dest.unlink(missing_ok=True)
        return f"Could not index that file: {type(error).__name__}", gr.update(), gr.update()

    # Clear the cached BM25 chunks so the new tender is searchable too.
    import rag.step4_retrieve as r4
    r4._chunks = None
    r4._bm25_cache.clear()

    choices = dropdown_choices() + [(dest.stem, dest.stem)]
    msg = (f"Added **{dest.name}** - {len(pages)} pages, {len(chunks)} chunks indexed. "
           f"Pick it in the dropdowns to query it. "
           f"(Run `python build_tender_cards.py` to give it a card.)")
    return msg, gr.update(choices=choices, value=dest.stem), gr.update(choices=choices)


def upload_company(file_obj):
    """Read a company profile from an uploaded .json or .txt file."""
    if file_obj is None:
        return gr.update(), "No file chosen."
    path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    try:
        raw = path.read_text()
    except Exception:
        return gr.update(), "Could not read that file as text."

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                inner = data.get("profile", data)
                raw = "\n".join(f"{k}: {v}" for k, v in inner.items())
        except json.JSONDecodeError:
            return gr.update(), "That file is not valid JSON."
    return gr.update(value=raw), f"Loaded {path.name}. Edit it below if you need to."


def ask(question, tender_choice):
    """Called when the user presses Ask. Returns answer, proof and a status line."""
    if not question.strip():
        return "Please type a question.", "", ""

    tender = None if tender_choice == ALL_TENDERS else tender_choice
    try:
        # The normal path is the LCEL chain from class. Only if that call fails
        # do we go through LiteLLM, which retries on a second model. Keeping it
        # this way round means litellm is not even imported unless something has
        # gone wrong, which matters on a 512 MB host.
        try:
            answer, chunks = generate_answer(question, tender=tender)
            model_used = CHAT_MODEL
        except Exception:
            answer, chunks, model_used = answer_with_fallback(question, tender=tender)
    except Exception as error:
        # Both models refused. On the Groq free tier this is almost always the
        # tokens-per-minute limit after a few questions in a row. Say so, rather
        # than showing the user a stack trace.
        name = type(error).__name__
        if "RateLimit" in name:
            return ("Both models are rate limited at the moment. The free Groq tier "
                    "allows 8000 tokens a minute and we have just used them. Wait "
                    "about a minute and ask again.", "", "")
        return (f"The model call failed ({name}). Check that GROQ_API_KEY is set "
                f"in .env and that there is a network connection.", "", "")

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

    with gr.Tab("Can we bid?"):
        gr.Markdown("### Check a company against a tender's eligibility rules")
        gr.Markdown(
            "This is the same RAG pipeline pointed at a different question. It "
            "retrieves the clauses about *who is allowed to bid*, puts them next to "
            "the company profile, and asks for a structured verdict instead of prose "
            "- so the result is a table with a page number against every line."
        )

        with gr.Row():
            elig_company = gr.Dropdown(
                label="Company", choices=company_choices(),
                value=company_choices()[0][1], scale=2,
            )
            elig_tender = gr.Dropdown(
                label="Tender", choices=dropdown_choices()[1:],
                value=dropdown_choices()[1][1], scale=3,
            )

        elig_profile = gr.Textbox(
            label="Company profile (edit it, or upload one below)",
            value=company_profile_text(company_choices()[0][1]),
            lines=12,
        )

        with gr.Accordion("Upload your own data", open=False):
            gr.Markdown(
                "**Company profile** - a `.json` or `.txt` file. A JSON object, or "
                "one with a `profile` key, is flattened into lines."
            )
            company_file = gr.File(label="Company profile", file_types=[".json", ".txt"])
            company_upload_msg = gr.Markdown("")

            gr.Markdown(
                "**Tender** - a PDF. It is chunked and added to the vector store "
                "straight away, so you can ask about a tender the system has never "
                "seen. It needs a real text layer; we did not add OCR."
            )
            tender_file = gr.File(label="Tender PDF", file_types=[".pdf"])
            tender_upload_msg = gr.Markdown("")

        elig_btn = gr.Button("Check eligibility", variant="primary")
        elig_out = gr.HTML()

        gr.Markdown(
            "This is a reading aid, not a legal opinion. It only sees the clauses "
            "retrieval found, which is why the report lists the pages it read - if a "
            "requirement is on a page that was not retrieved, it cannot judge it."
        )

        elig_company.change(
            fn=company_profile_text, inputs=[elig_company], outputs=[elig_profile]
        )
        elig_btn.click(
            fn=run_eligibility,
            inputs=[elig_company, elig_profile, elig_tender],
            outputs=[elig_out],
        )
        company_file.upload(
            fn=upload_company, inputs=[company_file],
            outputs=[elig_profile, company_upload_msg],
        )
        tender_file.upload(
            fn=upload_tender, inputs=[tender_file],
            outputs=[tender_upload_msg, elig_tender, tender_dropdown],
        )

    with gr.Tab("Why RAG?"):
        gr.Markdown("### The same model, the same question, with and without retrieval")
        gr.Markdown(
            "This is the question we expect to be asked: *how is this different from "
            "just asking ChatGPT?* Rather than argue about it, the app answers the "
            "same question twice with the same model, once with the retrieved tender "
            "clauses in the prompt and once without, and shows both."
        )
        with gr.Row():
            cmp_q = gr.Textbox(
                label="Question",
                value="What is the bid validity period?",
                lines=1, scale=3,
            )
            cmp_tender = gr.Dropdown(
                label="Search in", choices=dropdown_choices(),
                value=ALL_TENDERS, scale=2,
            )
        cmp_btn = gr.Button("Answer it both ways", variant="primary")
        cmp_out = gr.HTML()

        gr.Markdown(
            "**What to look for.** Sometimes the model refuses, which is harmless. "
            "The dangerous case is when it guesses: asked for the bid validity period "
            "of the NITI Aayog tender, it answered *\"typically 90 days\"*. The "
            "document says **75**. A bidder working to 90 days misses the deadline, "
            "and nothing in the answer warns them. Retrieval is what replaces a "
            "plausible guess with the clause and its page number."
        )

        cmp_btn.click(fn=compare_rag, inputs=[cmp_q, cmp_tender], outputs=[cmp_out])

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
            fn=lambda q, t: tour_start(q, t, ALL_TENDERS, DEFAULT_TENDER),
            inputs=[tour_q, tour_tender],
            outputs=[tour_step, tour_q, tour_panel, tour_label],
        )
        tour_next_btn.click(
            fn=lambda st, q, t: tour_move(st, q, t, 1, ALL_TENDERS, DEFAULT_TENDER),
            inputs=[tour_step, tour_q, tour_tender],
            outputs=[tour_step, tour_panel, tour_label],
        )
        tour_back_btn.click(
            fn=lambda st, q, t: tour_move(st, q, t, -1, ALL_TENDERS, DEFAULT_TENDER),
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
        gr.HTML(tour_html(CARDS))

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

    # When a host like Render runs this, it sets PORT and expects the app to
    # listen on every interface. Locally neither is set and Gradio uses its
    # own defaults.
    port = os.environ.get("PORT")

    # Gradio 6 takes the theme and the stylesheet here rather than on Blocks.
    demo.launch(
        theme=gr.themes.Soft(),
        css=CSS,
        share=args.share,
        server_name="0.0.0.0" if port else None,
        server_port=int(port) if port else None,
    )
