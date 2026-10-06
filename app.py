"""Gradio app: ask a question about a tender and get a grounded answer.

Run with:  python app.py

This file is only the user interface. All the RAG work happens in the
rag/ package, one file per step. Nothing in here does any retrieval or
any prompting itself - it just calls step 6, which calls steps 4 and 5.
"""

from pathlib import Path

import gradio as gr

from rag.step4_retrieve import TOP_K
from rag.step6_generate import answer_with_fallback

TENDER_FOLDER = Path(__file__).resolve().parent / "data" / "tenders"
ALL_TENDERS = "All tenders"

EXAMPLE_QUESTIONS = [
    "What is the earnest money deposit?",
    "What is the estimated cost put to bid?",
    "What is the last date and time for closing of bids?",
    "Who should the bid be addressed to?",
    "What happens to the EMD if a bidder withdraws the bid?",
]


def tender_choices():
    names = sorted(p.stem for p in TENDER_FOLDER.glob("*.pdf"))
    return [ALL_TENDERS] + names


def ask(question, tender_choice):
    """Called when the user presses Ask. Returns the answer and the sources."""
    if not question.strip():
        return "Please type a question.", "", ""

    tender = None if tender_choice == ALL_TENDERS else tender_choice
    answer, chunks, model_used = answer_with_fallback(question, tender=tender)

    # Show where the answer came from, so the user can check it in the PDF.
    seen = []
    for chunk in chunks:
        source = f"{chunk.metadata['tender']}, page {chunk.metadata['page']}"
        if source not in seen:
            seen.append(source)
    sources = "\n".join(f"- {s}" for s in seen)

    note = f"Retrieved {len(chunks)} chunks. Answered by: {model_used}"
    return answer, sources, note


with gr.Blocks(title="Tender RAG") as demo:
    gr.Markdown("## Tender RAG - ask a question about a tender document")
    gr.Markdown(
        "The answer is generated only from the tender PDFs in `data/tenders/`. "
        "If the tenders do not cover the question, the app says so instead of guessing."
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. What is the earnest money deposit?",
            lines=2,
            scale=3,
        )
        tender_dropdown = gr.Dropdown(
            label="Search in",
            choices=tender_choices(),
            value=ALL_TENDERS,
            scale=1,
        )

    ask_button = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=5)
    sources_box = gr.Textbox(label="Sources used (tender and page)", lines=5)
    note_box = gr.Textbox(label="Pipeline info", lines=1)

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_box)

    ask_button.click(
        fn=ask,
        inputs=[question_box, tender_dropdown],
        outputs=[answer_box, sources_box, note_box],
    )

if __name__ == "__main__":
    demo.launch()
