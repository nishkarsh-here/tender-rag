"""Step 3: turn chunks into embeddings and store them in a vector database.

In  : list of chunk Documents from step 2
Out : a Chroma vector store saved in chroma_db/

Why this step exists. We need to search by meaning, not by keyword.
If the user asks "what is the EMD?" but the tender says "earnest money
deposit", a keyword search finds nothing. An embedding model converts a
piece of text into a list of numbers (a vector) that represents its
meaning, and texts with similar meaning get vectors that point in a
similar direction. So we can compare meanings by comparing numbers.

The comparison itself is cosine similarity: the cosine of the angle
between two vectors. It runs from -1 (opposite) to 1 (identical in
direction). Chroma does this for us inside similarity_search.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# The embedding model. bge-small-en-v1.5 runs locally on the CPU and turns any
# text into a 384-number vector. We load it through fastembed, which runs the
# model with ONNX instead of PyTorch.
#
# We started with all-MiniLM-L6-v2 through sentence-transformers, which pulls
# in PyTorch - 552 MB on disk, too big for the free hosting tier we wanted to
# deploy on. Swapping to fastembed brought that down to 76 MB. We only kept the
# change because it also retrieved better: on the twelve hand-checked questions
# MiniLM found 11 and bge-small found 12, including the tender reference number
# that MiniLM had never managed.
#
# (The class notebook used nomic-embed-text through Ollama. Any of these work
# the same way from LangChain's side: embed_documents and embed_query.)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHROMA_FOLDER = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "tenders"


_embedding_model = None


def get_embedding_model():
    """Load the embedding model, once.

    Loading the model takes a second or two, and every retrieval needs it, so
    we keep the loaded model in a module-level variable and reuse it instead
    of building a new one on every call.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_model


def build_vector_store(chunks):
    """Embed every chunk and save the vectors to disk.

    This is the slow step, but we only run it once. Chroma writes the
    vectors, the chunk text and the metadata into chroma_db/, so the app
    can start instantly afterwards by calling load_vector_store().
    """
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_FOLDER,
    )

    # Clear anything indexed by a previous run, so re-running this file
    # rebuilds the index cleanly instead of storing every chunk twice.
    existing = store.get()["ids"]
    if existing:
        store.delete(ids=existing)

    store.add_documents(chunks)
    return store


def load_vector_store():
    """Open the vector store that build_vector_store() already wrote."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_FOLDER,
    )


if __name__ == "__main__":
    from rag.step1_load import load_all_tenders
    from rag.step2_chunk import split_into_chunks

    chunks = split_into_chunks(load_all_tenders())
    print(f"\nembedding {len(chunks)} chunks ...")

    store = build_vector_store(chunks)
    print(f"stored {len(store.get()['ids'])} chunks in {CHROMA_FOLDER}")

    # Show what an embedding actually looks like.
    vector = get_embedding_model().embed_query("earnest money deposit")
    print(f"\none embedding is a list of {len(vector)} numbers")
    print("first 8 numbers:", [round(v, 4) for v in vector[:8]])
