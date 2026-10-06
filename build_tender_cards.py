"""Build a short summary card for each tender, using the RAG pipeline itself.

Run this once after building the vector store:

    python build_tender_cards.py

It asks each tender a fixed set of questions through the same retrieval and
generation steps the app uses, and saves the answers to
data/tender_cards.json. The app reads that file so the home screen can show
what each tender actually is, instead of just a filename.

We cache the result because asking nine questions at start-up would make the
app slow to open. The cards are produced by the pipeline, not typed by hand.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from rag.step1_load import TENDER_FOLDER, load_tender_pdf
from rag.step3_embed_store import load_vector_store
from rag.step4_retrieve import retrieve_chunks
from rag.step5_prompt import format_context
from rag.step6_generate import CHAT_MODEL, MAX_TOKENS, TEMPERATURE

load_dotenv(override=True)

CARDS_FILE = Path(__file__).resolve().parent / "data" / "tender_cards.json"


class TenderCard(BaseModel):
    """The handful of facts a bidder looks for first."""

    title: str = Field(description="What is being purchased or built, in under 10 words")
    organisation: str = Field(description="The organisation inviting the tender")
    reference: str = Field(description="The tender or NIT reference number, or 'Not stated'")
    closing: str = Field(description="Last date (and time if given) for submitting the bid, or 'Not stated'")
    emd: str = Field(description="The earnest money deposit amount with currency, or 'Not stated'")
    value: str = Field(description="Estimated cost or value of the work, or 'Not stated'")


# We retrieve with one targeted query per field rather than one long question.
# A single broad question embeds to an average of several topics and tends to
# miss the page that actually holds the answer - the same effect that made
# large chunks perform badly. Several specific queries, pooled, work better.
LOOKUP_QUERIES = [
    "subject of the tender, name of work, brief description of the item to be purchased",
    "tender reference number, NIT number, file number",
    "last date and time for closing of bids, bid end date",
    "earnest money deposit EMD amount in rupees",
    "estimated cost put to bid, value of the work",
    "name of the organisation, department and office inviting the tender",
]

EXTRACT_INSTRUCTION = (
    "From these extracts, identify what is being purchased or maintained, the "
    "organisation inviting the tender, the reference number, the last date for "
    "submission, the earnest money deposit, and the estimated cost."
)


def cover_pages(tender_name, upto=2):
    """Every chunk from the first couple of pages of a tender.

    A tender states the name of the work, the reference number and the key
    dates on its cover page, almost always in a summary table. Retrieval alone
    sometimes ranks that table low, because it mixes many topics in one chunk.
    Since we know where it is, we just include it.
    """
    store = load_vector_store()
    records = store.get(where={"tender": tender_name})
    pairs = sorted(
        (
            (meta["page"], meta.get("start_index", 0), text, meta)
            for text, meta in zip(records["documents"], records["metadatas"])
            if meta["page"] <= upto
        ),
        key=lambda row: (row[0], row[1]),
    )
    return [Document(page_content=text, metadata=meta) for _, _, text, meta in pairs]


def build_card(tender_name):
    """Ask the pipeline for this tender's key facts, as a structured object."""
    # Start with the cover pages, then pool the chunks each query retrieves,
    # keeping every chunk only once.
    chunks, seen = [], set()
    for chunk in cover_pages(tender_name) :
        key = (chunk.metadata["page"], chunk.page_content[:60])
        seen.add(key)
        chunks.append(chunk)

    for query in LOOKUP_QUERIES:
        for chunk in retrieve_chunks(query, k=4, tender=tender_name):
            key = (chunk.metadata["page"], chunk.page_content[:60])
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)

    llm = init_chat_model(
        CHAT_MODEL,
        model_provider="groq",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    # with_structured_output makes the model return the fields of TenderCard
    # directly, so we do not have to parse prose. Any field the extracts do
    # not cover comes back as "Not stated".
    structured_llm = llm.with_structured_output(TenderCard)

    prompt = (
        "Use only these extracts from a government tender document.\n"
        "If a field is not stated in the extracts, write exactly 'Not stated'.\n\n"
        f"{format_context(chunks)}\n\n{EXTRACT_INSTRUCTION}"
    )
    return structured_llm.invoke(prompt).model_dump()


def main():
    cards = []
    for pdf_path in sorted(Path(TENDER_FOLDER).glob("*.pdf")):
        name = pdf_path.stem
        print(f"reading {name} ...")

        pages = load_tender_pdf(pdf_path)
        card = build_card(name)
        card["tender"] = name
        card["pages"] = len(pages)
        card["file"] = pdf_path.name
        cards.append(card)

        print(f"   {card['title']} | {card['organisation']} | EMD {card['emd']}")

    CARDS_FILE.write_text(json.dumps(cards, indent=2))
    print(f"\nwrote {len(cards)} cards to {CARDS_FILE}")


if __name__ == "__main__":
    main()
