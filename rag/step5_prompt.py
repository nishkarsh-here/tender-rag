"""Step 5: build the prompt that puts the retrieved chunks around the question.

In  : the question, and the chunks retrieved in step 4
Out : a finished prompt ready to send to the LLM

Why this step exists. This is the "Augmented" in Retrieval Augmented
Generation. On its own the model has never seen these tenders, so it would
either refuse or invent an answer. Here we paste the retrieved clauses into
the prompt as {context} and tell the model to answer only from them.

Two instructions in the template matter a lot:
  - answer only from the context, so the model cannot fall back on its own
    general knowledge and make up an EMD amount
  - if the context does not contain the answer, say so. For a tender, a
    confident wrong number is far worse than "this is not stated".
"""

from langchain_core.prompts import ChatPromptTemplate

TEMPLATE = """You are helping a bidder read a government tender document.

Answer the question using ONLY the tender extracts given below.

Rules:
- If the extracts do not contain the answer, reply exactly:
  "This is not stated in the tender extracts I was given."
- Do not use any knowledge from outside the extracts.
- Quote the figure or the wording from the tender where you can.
- Keep the answer short, at most 4 sentences.

Tender extracts:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = ChatPromptTemplate.from_template(TEMPLATE)


def format_context(chunks):
    """Turn the retrieved chunks into one labelled block of text.

    Each chunk is labelled with its tender and page so the model can refer
    to them, and so a human reading the prompt can check where a sentence
    in the answer came from.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[Extract {i} | {chunk.metadata['tender']} | page {chunk.metadata['page']}]"
        parts.append(f"{label}\n{chunk.page_content}")
    return "\n\n".join(parts)


def build_prompt(question, chunks):
    """Fill the template with the context and the question."""
    return RAG_PROMPT.format(context=format_context(chunks), question=question)


if __name__ == "__main__":
    from rag.step4_retrieve import retrieve_chunks

    question = "What is the earnest money deposit?"
    chunks = retrieve_chunks(question, k=2)

    print("=" * 70)
    print("THIS IS THE EXACT TEXT THAT GETS SENT TO THE LLM")
    print("=" * 70)
    print(build_prompt(question, chunks))
