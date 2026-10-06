"""Step 4: find the chunks that are closest in meaning to the question.

In  : a question in plain English
Out : the TOP_K most similar chunks, with their tender name and page

Why this step exists. This is the "Retrieval" in Retrieval Augmented
Generation. The question is embedded with the same model used in step 3,
then Chroma compares that vector against all 142 stored chunk vectors and
returns the closest ones. Only those chunks go into the prompt, so the
model reads a page of relevant text instead of a 30-page tender.
"""

from rag.step3_embed_store import load_vector_store

# How many chunks to retrieve. This is a hyperparameter that can be tuned.
#
# Our chunks are small (300 characters), so one chunk on its own is often
# only part of a clause. We tested the same six hand-checked questions at
# different values and needed k=8 before all six answers appeared in the
# retrieved text:
#
#     k=4  ->  5 of 6
#     k=6  ->  5 of 6
#     k=8  ->  6 of 6
#
# 8 chunks of 300 characters is about 2400 characters of context, which is
# still a shorter prompt than 4 chunks of 1000 would have been.
TOP_K = 8


def retrieve_chunks(question, k=TOP_K, tender=None):
    """Return the k chunks most similar to the question.

    If `tender` is given, only chunks from that tender are searched. This is
    metadata filtering: the search is still by meaning, but it is restricted
    to the documents whose metadata matches.
    """
    store = load_vector_store()

    search_kwargs = {"k": k}
    if tender:
        search_kwargs["filter"] = {"tender": tender}

    retriever = store.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(question)


def retrieve_with_scores(question, k=TOP_K):
    """Same as retrieve_chunks, but also returns the distance score.

    Only used for showing the numbers in the notebook. A lower distance
    means the chunk is closer in meaning to the question.
    """
    store = load_vector_store()
    return store.similarity_search_with_score(question, k=k)


if __name__ == "__main__":
    question = "What is the earnest money deposit for the lift tender?"
    print(f"Question: {question}\n")

    for i, (chunk, score) in enumerate(retrieve_with_scores(question), start=1):
        source = f"{chunk.metadata['tender']} p.{chunk.metadata['page']}"
        print(f"--- chunk {i}  [{source}]  distance={score:.4f} ---")
        print(chunk.page_content[:300].replace("\n", " "))
        print()
