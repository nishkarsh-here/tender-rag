# Teacher Walkthrough

How we present this project, and the answers we should be able to give without
opening three files first.

Format: **Question → Answer → Where to look → What to show.**

---

## The 8-minute demo plan

| Time | What we do |
|---|---|
| 0:00–1:00 | The problem: a 25-page tender, and the handful of facts a bidder must not miss |
| 1:00–2:00 | The pipeline, using the `rag/` folder listing as the diagram |
| 2:00–5:30 | Live run in the notebook: load → clean → chunk → embed → retrieve → prompt → answer |
| 5:30–6:30 | The Gradio app, including an out-of-context question that gets refused |
| 6:30–7:15 | Break the primary model, show the LiteLLM fallback answer |
| 7:15–8:00 | What we chose and why (chunk size, k, temperature), and the limitations |

The single most convincing thing to show is the cell that prints the **finished
prompt** with the retrieved tender clauses inside it. Everything else follows
from that.

---

## "Show me your RAG."

**Answer.** The six files in `rag/` are the six stages of the pipeline, in order.

**Show.** The folder listing:

```
rag/step1_load.py          load the PDF, clean the text
rag/step2_chunk.py         cut into chunks
rag/step3_embed_store.py   embed and store
rag/step4_retrieve.py      similarity search
rag/step5_prompt.py        build the prompt
rag/step6_generate.py      call the LLM
```

---

## Loading and cleaning

**Where is the document loaded?**
`rag/step1_load.py`, `load_tender_pdf()`. We use `PyPDFLoader`, which returns one
LangChain `Document` per page. We keep pages separate because the page number
travels with the text all the way to the final answer and becomes the citation.

**Where is text extracted and cleaned?**
Same file, `clean_text()`. Show the before/after cell in the notebook.

**Why did you need cleaning?**
Two of our tenders put every single word on its own line, because each word is a
separately positioned block in a table. Raw, a chunk would be a column of words
instead of a sentence, and retrieval got much worse. `clean_text()` detects that
by average line length and reflows the text.

---

## Chunking

**Where is chunking happening?**
`rag/step2_chunk.py`, `split_into_chunks()`.

**Why do you need chunking?**
Two reasons. A tender is far longer than the model's context window. And even if
it fitted, handing the model 25 pages buries the one line that answers the
question in thousands of irrelevant lines. Smaller pieces let us retrieve only
the relevant part.

**Why `RecursiveCharacterTextSplitter`?**
It tries paragraph breaks first, then single newlines, then spaces, then
characters, and only moves to a smaller separator when a piece is still too big.
That keeps related text together better than cutting blindly every N characters.

**Why chunk size 300?**
We did not take the default. We started at 1000, the value used in class, and it
performed badly. Page 1 of a tender is a summary table holding many facts at
once — NIT number, dates, estimated cost, EMD — so at 1000 characters the whole
table becomes one chunk, its embedding is an average of many topics, and it
matches no single question strongly. We tested six questions whose answers we
had checked by hand in the PDFs:

| chunk size | answers found |
|---|---|
| 300 | 5 of 6 |
| 500 | 3 of 6 |
| 800 | 3 of 6 |
| 1000 | 1 of 6 |
| 1500 | 2 of 6 |

**Show.** The tuning cell in the notebook — it re-runs that table live.

**Why overlap 60?**
About 20% of the chunk size, the same ratio as the class notebook. Neighbouring
chunks share their edges, so a clause sitting exactly on a chunk boundary still
appears complete in one of them.

**What does `add_start_index` do?**
Records where the chunk started inside its page, so a chunk can be traced back to
its exact position in the original text.

---

## Embeddings

**Where are embeddings generated?**
`rag/step3_embed_store.py`, `get_embedding_model()`.

**Why are embeddings required?**
Because we need to search by meaning, not by keyword. One tender says "EMD",
another says "earnest money deposit", a third says "bid security". A keyword
search for "EMD" misses two of them. An embedding turns text into a vector that
represents its meaning, and similar meanings get vectors pointing in similar
directions.

**How are the question and the chunks compared?**
Cosine similarity — the cosine of the angle between the two vectors, running
from -1 to 1. Chroma does it internally during `similarity_search`.

**Show.** The notebook cell that computes cosine similarity by hand in NumPy.
"last date for submission of bids" vs "closing of bids is 11-05-2026" scores
0.73 despite sharing almost no words; an unrelated sentence scores near zero.

Be ready for the follow-up: that same cell shows "earnest money deposit" vs
"EMD" scoring only 0.26. Say so plainly — a small embedding model does not
reliably link an abbreviation to its full form. Retrieval still works because
0.26 is far above an unrelated sentence, but it is a genuine weakness, and it
is exactly why hybrid retrieval with BM25 is on our improvements list.

**Why this embedding model?**
`all-MiniLM-L6-v2`. It runs locally on the CPU, needs no API key and no separate
model server, produces a 384-number vector, and indexes all three tenders in
seconds. The class notebook used `nomic-embed-text` through Ollama; the LangChain
interface is the same, and this version runs with nothing but `pip install`.

**Can the embedding model have a different dimension from the chat model?**
Yes — they are unrelated models doing unrelated jobs. The only thing that must
match is the embedding model used at indexing time and at query time. If those
differed, the vectors would not be comparable.

---

## Storage and retrieval

**Where are embeddings stored?**
`rag/step3_embed_store.py`, `build_vector_store()` — a Chroma collection
persisted to `chroma_db/`.

**Why Chroma rather than FAISS?**
Class covered both. Chroma keeps the vectors, the chunk text and the metadata
together in one collection and persists with a single `persist_directory`
argument. FAISS needs a separate docstore and an index-to-id mapping — more
moving parts than this project needs.

**Where is similarity search / retrieval happening?**
`rag/step4_retrieve.py`, `retrieve_chunks()`, via
`store.as_retriever(search_kwargs={"k": 8})`.

**What exactly is retrieved?**
LangChain `Document` objects. Each has `page_content` (the chunk text) and
`metadata` holding the tender name, the page number and the start index. The
metadata is what makes the citation possible.

**Why k = 8?**
Our chunks are small, so one chunk is often only part of a clause. On the same
six hand-checked questions: k=4 gave 5 of 6, k=6 gave 5 of 6, k=8 gave 6 of 6.
Eight chunks of 300 characters is about 2400 characters — still a shorter prompt
than four chunks of 1000 would have been.

**What is metadata filtering for?**
We indexed three tenders, and "what is the EMD?" matches all three because all
three discuss EMD. Passing `tender=` restricts the search to one document while
still searching by meaning.

**Show.** The notebook cell that runs the same question with no filter, then with
each filter, and prints how the sources change.

---

## The prompt

**Where is the prompt built?**
`rag/step5_prompt.py` — `RAG_PROMPT` is the `ChatPromptTemplate`, and
`build_prompt()` fills it in.

**Where is the retrieved context inserted?**
Into the `{context}` placeholder. `format_context()` labels each chunk with its
tender and page before pasting it in.

**Show. (Most important thing in the demo.)** The notebook cell that prints the
finished prompt. You can read the tender clauses sitting inside the prompt, then
read the answer, and check that one came from the other.

**How is this different from just asking ChatGPT?**
ChatGPT has never seen this tender. Asked for the EMD it would either refuse or
invent a number. We retrieve the actual clause from this specific PDF, put it in
the prompt, and the answer cites the page it came from. The model is doing
reading comprehension on text we supplied, not recalling facts.

**What happens if the answer is not in the document?**
The template instructs the model to reply "This is not stated in the tender
extracts I was given." For a tender this matters a lot — a bidder acting on an
invented EMD figure loses the bid, so refusing is the correct behaviour.

**Show.** Ask "Who is the CEO of Microsoft?" in the app. It refuses.

---

## The LLM call

**Where is the LLM called?**
`rag/step6_generate.py`, `generate_answer()`.

**What does the chain look like?**
The LCEL chain from class:

```python
chain = RAG_PROMPT | llm | StrOutputParser()
```

The model is created with `init_chat_model("openai/gpt-oss-120b",
model_provider="groq")`.

**What temperature, and why?**
0. Reading a tender is factual extraction, not creative writing — the same
question should give the same answer every time. The class notebook on
generation parameters puts deterministic, extraction-style tasks at the low end.

**Why is `max_tokens` set to 1200?**
The gpt-oss models reason before answering, and that reasoning counts against the
token budget. With too small a budget the whole allowance is spent thinking and
the reply comes back empty. We hit this and raised the limit.

---

## LiteLLM and the fallback

**Where is LiteLLM used?**
`rag/step6_generate.py`, `ask_llm_with_fallback()` — one function, about twelve
lines.

**What happens when the first model fails?**
The exception is caught, the same prompt is sent to the fallback model, and the
function returns both the answer and the name of the model that produced it, so
the app can display it.

```
primary   groq/openai/gpt-oss-120b
              |  (call raises)
              v
fallback  groq/openai/gpt-oss-20b
```

**Why did you add it?**
A free API tier can rate-limit in the middle of a demo. It is cheap insurance,
and it means a model outage degrades the answer rather than killing the app.

**Show.** In the notebook, set `PRIMARY_MODEL` to a model that does not exist and
re-run. The failure prints, the fallback answers, and the reported model changes.

**Why LiteLLM when class taught `init_chat_model`?**
Both are in the file, side by side. LangChain is the main path and is what the
class taught. LiteLLM gives one uniform call signature across providers, which
makes the fallback a four-line try/except instead of provider-specific handling.
We kept the class path as the default and use LiteLLM for resilience.

---

## About the project as a whole

**Which concepts from class are demonstrated here?**
Document loading, text cleaning, chunking with
`RecursiveCharacterTextSplitter`, embeddings, cosine similarity, a vector store,
`as_retriever` with top-k, metadata filtering, `ChatPromptTemplate`,
`init_chat_model`, generation parameters, the LCEL chain with
`StrOutputParser`, and grounding with an out-of-context test. The README has the
concept-to-file table.

**Why no agents?**
The assignment says the project should be on RAG and that agents are optional. We
put the effort into getting the RAG right — the chunking experiment, the metadata
filtering, the grounding behaviour — rather than adding orchestration the problem
does not need.

**What are the limitations?**
One tender is a scan whose OCR text has errors, which hurts retrieval on that
document. We judged retrieval with six hand-checked questions, which is a sanity
check and not a labelled evaluation. Only PDFs with a text layer work. If the
right clause is not in the top 8, the system says it does not know rather than
finding it.

**What was hardest?**
Retrieval, not generation. The model was never the problem. Getting the right
chunk in front of it was, and that is what the chunk-size experiment is about.

**What did AI tools help with?**
Answer this honestly and make sure it matches the appendix slide. Be specific
about what was generated, what was edited, and what was written by hand, and be
ready to explain any line in the repository.
