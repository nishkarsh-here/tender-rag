"""Step 4: find the chunks that are closest to the question.

In  : a question in plain English
Out : the TOP_K best chunks, with their tender name and page

This is the "Retrieval" in Retrieval Augmented Generation. Only the chunks
found here go into the prompt, so the model reads a page of relevant text
instead of a 30-page tender.

We use semantic search: the question and the chunks are embedded with the same
model, and Chroma returns the chunks whose vectors are closest.

We also tried the hybrid BM25 + semantic retriever from class. It did not help
on our data, and we kept the simpler one. The measurements are in
WHY_NOT_HYBRID below, and retrieve_hybrid() is still here so the comparison can
be re-run.
"""

from langchain_community.retrievers import BM25Retriever

# EnsembleRetriever lived in langchain.retrievers in the version used in class.
# It moved in the newer release, so we try the new path first and fall back.
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # older langchain, as used in the class notebook
    from langchain.retrievers import EnsembleRetriever

from rag.step1_load import load_all_tenders
from rag.step2_chunk import split_into_chunks
from rag.step3_embed_store import load_vector_store

# How many chunks to retrieve. This is a hyperparameter that can be tuned.
#
# Our chunks are small (300 characters), so one chunk is often only part of a
# clause and we need several. Measured on twelve questions whose answers we
# looked up by hand in the PDFs:
#
#     k=8   10 / 12
#     k=10  11 / 12
#     k=12  11 / 12
#     k=20  12 / 12
#
# We chose 10. Going to 20 does find the last one, but it doubles the amount of
# text in the prompt to chase a single tender reference number, and a longer
# prompt gives the model more irrelevant clauses to wade through.
TOP_K = 10

# Weights used by retrieve_hybrid(), kept for the comparison below.
BM25_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5

WHY_NOT_HYBRID = """
The class notebook covers hybrid retrieval: BM25 for exact keywords, combined
with semantic search through EnsembleRetriever. We implemented it, because
semantic search at k=8 was missing "Who should the demand draft be drawn in
favour of?" even though page 1 says "Demand Draft drawn in favour of The
Director". BM25 matches that phrase directly, so hybrid looked like the fix.

On our first test of ten questions hybrid did win, 9/10 against 7/10. But we
had chosen those questions after watching semantic search fail, so the test was
biased towards the thing we had just built. On a fairer set of twelve
questions, written before running either retriever:

    semantic only, k=8        10 / 12
    BM25 only, k=8             4 / 12
    hybrid, 8 each, keep 8     8 / 12
    hybrid, 8 each, keep 10   10 / 12
    semantic only, k=10       11 / 12

Hybrid fixed the demand draft question and broke two others, because BM25
pulled in chunks that merely repeated a keyword and pushed out the summary
table that actually held the answer. Simply raising k on semantic search did
better, and is a great deal simpler.

So we kept semantic search. This is the honest result, not the one we expected.
"""

# Loading and chunking the PDFs takes a second or two, and BM25 needs the
# chunks in memory, so we do it once and keep the result.
_chunks = None
_bm25_cache = {}


def all_chunks():
    """Every chunk of every tender, loaded once per run."""
    global _chunks
    if _chunks is None:
        _chunks = split_into_chunks(load_all_tenders())
    return _chunks


def get_bm25(tender=None):
    """A BM25 keyword retriever, over one tender or over all of them."""
    if tender not in _bm25_cache:
        chunks = all_chunks()
        if tender:
            chunks = [c for c in chunks if c.metadata["tender"] == tender]
        retriever = BM25Retriever.from_documents(chunks)
        retriever.k = TOP_K
        _bm25_cache[tender] = retriever
    return _bm25_cache[tender]


def get_semantic(k=TOP_K, tender=None):
    """A semantic retriever backed by the Chroma vector store.

    Passing `tender` adds a metadata filter, so the search is still by meaning
    but only over the chunks belonging to that one document.
    """
    search_kwargs = {"k": k}
    if tender:
        search_kwargs["filter"] = {"tender": tender}
    return load_vector_store().as_retriever(search_kwargs=search_kwargs)


def retrieve_chunks(question, k=TOP_K, tender=None):
    """Return the k chunks closest in meaning to the question.

    This is what the app and the notebook use.
    """
    return get_semantic(k, tender).invoke(question)


def retrieve_hybrid(question, k=TOP_K, tender=None):
    """BM25 and semantic search combined. Not used by default - see WHY_NOT_HYBRID.

    EnsembleRetriever runs both retrievers and merges their rankings with
    reciprocal rank fusion, so a chunk both methods rank highly comes out on
    top. Kept so the comparison in the notebook can be re-run.
    """
    ensemble = EnsembleRetriever(
        retrievers=[get_bm25(tender), get_semantic(k, tender)],
        weights=[BM25_WEIGHT, SEMANTIC_WEIGHT],
    )
    return ensemble.invoke(question)[:k]


def retrieve_with_scores(question, k=TOP_K):
    """Semantic search with the distance score, for showing the numbers.

    A lower distance means the chunk is closer in meaning to the question.
    Only semantic search produces a distance, which is why this uses it.
    """
    return load_vector_store().similarity_search_with_score(question, k=k)


if __name__ == "__main__":
    question = "Who should the demand draft be drawn in favour of?"
    print(f"Question: {question}  (TOP_K = {TOP_K})")
    print(WHY_NOT_HYBRID)

    for label, docs in [
        ("SEMANTIC (what we use)", retrieve_chunks(question, tender="nitt_glove_box")),
        ("HYBRID (tried, not kept)", retrieve_hybrid(question, tender="nitt_glove_box")),
    ]:
        print(f"--- {label} ---")
        for i, chunk in enumerate(docs, start=1):
            marker = "  <-- has the answer" if "favour of The Director" in chunk.page_content else ""
            print(f"{i}. page {chunk.metadata['page']}: "
                  f"{chunk.page_content[:85]}".replace("\n", " ") + marker)
        print()
