"""Content for the Guided tour and Tech stack tabs.

This is here rather than in app.py so that app.py stays what it should be: the
layout of the page and the buttons. Nothing in this file does any retrieval of
its own, it calls the same six steps the rest of the project uses.
"""

import html

from langchain_community.document_loaders import PyPDFLoader

from rag.step1_load import TENDER_FOLDER, clean_text, load_tender_pdf
from rag.step2_chunk import CHUNK_OVERLAP, CHUNK_SIZE, split_into_chunks
from rag.step3_embed_store import EMBEDDING_MODEL, get_embedding_model, load_vector_store
from rag.step4_retrieve import TOP_K, retrieve_chunks
from rag.step5_prompt import build_prompt
from rag.step6_generate import (
    CHAT_MODEL,
    FALLBACK_MODEL,
    PRIMARY_MODEL,
    TEMPERATURE,
    answer_with_fallback,
)

TOUR_STEPS = [
    ("1. Load the PDF", "rag/step1_load.py", ["LangChain", "PyPDFLoader", "pypdf", "regex"]),
    ("2. Split into chunks", "rag/step2_chunk.py", ["LangChain", "RecursiveCharacterTextSplitter"]),
    ("3. Embed and store", "rag/step3_embed_store.py", ["sentence-transformers", "HuggingFaceEmbeddings", "Chroma"]),
    ("4. Retrieve", "rag/step4_retrieve.py", ["Chroma", "as_retriever", "cosine similarity"]),
    ("5. Build the prompt", "rag/step5_prompt.py", ["LangChain", "ChatPromptTemplate"]),
    ("6. Generate the answer", "rag/step6_generate.py", ["init_chat_model", "LCEL", "Groq", "LiteLLM"]),
]

def tour_html(cards):
    """The guided tour: what each stage does, and what it is built with.

    The numbers are read from the code and the live vector store, so this page
    cannot drift out of date with what the system actually does.
    """
    try:
        indexed = len(load_vector_store().get()["ids"])
    except Exception:
        indexed = "not built yet"

    pages = sum(c["pages"] for c in cards) if cards else "?"

    steps = [
        (
            "rag/step1_load.py",
            "Read each tender PDF into one Document per page, then clean the text. "
            "Keeping pages separate is what lets an answer cite a page number later. "
            "Two of our tenders put every word on its own line, so the cleaner detects "
            "that and reflows them into sentences.",
            ["LangChain", "PyPDFLoader", "pypdf", "re (regex)"],
            f"{len(cards)} tenders, {pages} pages with text",
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


def run_tour_stage(step, question, tender, all_tenders_label, default_tender):
    """Produce the display for one stage, actually running that stage."""
    title, file, tech = TOUR_STEPS[step]
    head = (
        f'<div class="tour-head"><div><span class="tour-stage">Stage {step + 1} of 6</span>'
        f'<div class="tour-title">{html.escape(title)}</div>'
        f'<div class="tour-file">{html.escape(file)}</div></div></div>' + _chips(tech)
    )

    tender_name = None if tender == all_tenders_label else tender
    sample = tender_name or default_tender

    if step == 0:
        pages = load_tender_pdf(TENDER_FOLDER / f"{sample}.pdf")
        raw = PyPDFLoader(str(TENDER_FOLDER / f"{sample}.pdf")).load()[0].page_content
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
        pages = load_tender_pdf(TENDER_FOLDER / f"{sample}.pdf")
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
            "<p><b>Three chunks in a row.</b> Notice the overlap at their edges:</p>" + shown
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
            "scores high. Note the abbreviation in the middle row scoring poorly. A "
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


def tour_start(question, tender, all_tenders_label, default_tender):
    q = question.strip() or "What is the earnest money deposit?"
    panel, label = run_tour_stage(0, q, tender, all_tenders_label, default_tender)
    return 0, q, panel, label


def tour_move(step, question, tender, delta, all_tenders_label, default_tender):
    step = max(0, min(len(TOUR_STEPS) - 1, step + delta))
    panel, label = run_tour_stage(step, question, tender, all_tenders_label, default_tender)
    return step, panel, label
