"""Step 2: cut the pages into smaller overlapping chunks.

In  : list of page Documents from step 1
Out : list of chunk Documents, each still knowing its tender and page

Why this step exists. Two reasons:
  1. A tender runs to 20-30 pages. We cannot put all of it in the prompt,
     because the model has a limited context window.
  2. Even if we could, it would hurt. If we hand the model everything, the
     one line that answers the question is buried in thousands of irrelevant
     lines. Smaller pieces let us retrieve only the relevant part.

So we cut the document into chunks, and later we search over chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Both of these are hyperparameters that can be tuned.
#
# CHUNK_SIZE = 300 characters. We started at 1000, which is the default in
# the class RAG notebook, and it worked badly here. The reason is that the
# first page of a tender is a summary table holding many different facts at
# once - NIT number, dates, estimated cost, EMD. At 1000 characters that
# whole table becomes one chunk, so its embedding is an average of many
# topics and it does not match any single question strongly.
#
# We tested this on six questions whose answers we had checked by hand in
# the PDFs (see notebooks/01_rag_pipeline_explained.ipynb):
#
#     chunk_size   chunks   answers found in the retrieved text
#            300      441   5 of 6
#            500      275   3 of 6
#            800      171   3 of 6
#           1000      142   1 of 6
#           1500      100   2 of 6
#
# Smaller chunks split that table into separate rows, so each chunk is about
# one thing and its embedding is specific. 300 was clearly the best here.
#
# CHUNK_OVERLAP = 60 characters, kept at about 20% of the chunk size, same
# ratio as the class notebook. Neighbouring chunks share their edges, so a
# clause sitting on a chunk boundary still appears complete in one of them.
CHUNK_SIZE = 300
CHUNK_OVERLAP = 60


def split_into_chunks(documents):
    """Split page Documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first,
    then single newlines, then spaces, then individual characters. It only
    moves to a smaller separator when a piece is still too big. That is why
    it is "recursive", and it is why it keeps related text together better
    than cutting blindly every 1000 characters.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # records where the chunk began in the page
    )
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    from rag.step1_load import load_all_tenders

    pages = load_all_tenders()
    chunks = split_into_chunks(pages)

    print(f"\n{len(pages)} pages  ->  {len(chunks)} chunks")
    sizes = [len(c.page_content) for c in chunks]
    print(f"chunk size: smallest {min(sizes)}, largest {max(sizes)}, "
          f"average {sum(sizes) // len(sizes)}")

    print("\n--- one example chunk ---")
    example = chunks[3]
    print("metadata:", example.metadata)
    print(example.page_content[:500])
