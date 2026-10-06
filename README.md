# Tender RAG

A small Retrieval Augmented Generation system that answers questions about
government tender documents, and shows which page each answer came from.

Generative AI, Assignment 3 — AI & Data Science Program, Jio Institute.

**Live demo:** https://70ea21f7443a868304.gradio.live

That is a Gradio share link, so it is only up while we are running the app on
our machine, and it expires after about a week. If it does not open, the app
runs locally in two commands — see [How to run](#how-to-run) below.

---

## The problem

A government tender is 20–30 pages of dense clauses. Before bidding, a company
has to pull out a handful of facts: the earnest money deposit, the last date for
submission, the estimated cost, who the bid is addressed to, what happens if the
bid is withdrawn.

Those facts are scattered across the document and worded inconsistently — one
tender says "EMD", another says "earnest money deposit", a third says "bid
security". Missing one of them can get a bid rejected outright.

Reading every tender by hand is slow, and a plain keyword search (Ctrl+F) fails
as soon as the wording differs from what you typed.

## Why RAG

We cannot just ask an LLM, because the model has never seen this specific tender
and would invent an answer. For a tender, a confident wrong number is worse than
no answer at all.

RAG fixes this. We index the tender ourselves, retrieve only the clauses that
relate to the question, and put those clauses into the prompt. The model then
answers from the actual document, and we can show the page it came from.

Embeddings are what make the retrieval work across wording differences: "EMD"
and "earnest money deposit" end up close together in vector space, so the right
clause is found even though no word matches.

## How it works

```
   data/tenders/*.pdf
          |
          v   step1_load.py        read the PDF, clean the text
          v   step2_chunk.py       cut into 300-character overlapping chunks
          v   step3_embed_store.py embed each chunk, save to Chroma
          |
   [ user question ]
          |
          v   step4_retrieve.py    find the 8 closest chunks
          v   step5_prompt.py      put those chunks into the prompt
          v   step6_generate.py    call the LLM (LiteLLM fallback if it fails)
          |
          v
   answer + the tender and page it came from
```

## Folder structure

| Path | What it is |
|---|---|
| `rag/step1_load.py` | Loads the tender PDF and cleans the extracted text |
| `rag/step2_chunk.py` | Splits pages into overlapping chunks |
| `rag/step3_embed_store.py` | Creates embeddings and stores them in Chroma |
| `rag/step4_retrieve.py` | Finds the chunks closest to the question |
| `rag/step5_prompt.py` | Builds the prompt that holds the retrieved context |
| `rag/step6_generate.py` | Calls the LLM; also holds the LiteLLM fallback |
| `app.py` | Gradio interface. Contains no RAG logic itself |
| `notebooks/01_rag_pipeline_explained.ipynb` | Runs the pipeline step by step and prints each stage |
| `data/tenders/` | Three real public tender PDFs |
| `chroma_db/` | The saved vector store (created by step 3) |

Each file in `rag/` does one stage and nothing else. You can run any of them on
its own to see what that stage produces:

```bash
python -m rag.step1_load
python -m rag.step2_chunk
python -m rag.step4_retrieve
python -m rag.step5_prompt
```

## Where each concept is implemented

| Concept | File | Function |
|---|---|---|
| Document loading | `rag/step1_load.py` | `load_tender_pdf()` |
| Text cleaning | `rag/step1_load.py` | `clean_text()` |
| Chunking | `rag/step2_chunk.py` | `split_into_chunks()` |
| Embeddings | `rag/step3_embed_store.py` | `get_embedding_model()` |
| Vector store | `rag/step3_embed_store.py` | `build_vector_store()` |
| Similarity search / retrieval | `rag/step4_retrieve.py` | `retrieve_chunks()` |
| Metadata filtering | `rag/step4_retrieve.py` | `retrieve_chunks(tender=...)` |
| Prompt template | `rag/step5_prompt.py` | `RAG_PROMPT`, `build_prompt()` |
| Inserting retrieved context | `rag/step5_prompt.py` | `format_context()` |
| LLM call (LCEL chain) | `rag/step6_generate.py` | `generate_answer()` |
| Generation parameters | `rag/step6_generate.py` | `TEMPERATURE`, `MAX_TOKENS` |
| LiteLLM + fallback model | `rag/step6_generate.py` | `ask_llm_with_fallback()` |

## Setup

Needs Python 3.12 and a free Groq API key from https://console.groq.com/keys

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then put your Groq key in .env
```

## How to run

Build the vector store once (reads the PDFs, embeds the chunks):

```bash
python -m rag.step3_embed_store
```

Then start the app:

```bash
python app.py
```

It opens at http://127.0.0.1:7860

To see the whole pipeline stage by stage instead, open
`notebooks/01_rag_pipeline_explained.ipynb`.

## Example

Question: *What is the earnest money deposit for the home lift tender?*

Answer:

> The Earnest Money Deposit (EMD) required is **Rs. 56,000/- (Rupees Fifty-Six
> Thousand Only)**.

Sources: `iitpkd_lift, page 2`

Question: *Who is the CEO of Microsoft?*

Answer:

> This is not stated in the tender extracts I was given.

## The LiteLLM fallback

The LLM call can fail — most often because a free API tier rate-limits in the
middle of a demo. `ask_llm_with_fallback()` in `rag/step6_generate.py` tries a
primary model, and if that call raises, it sends the same prompt to a second
model and reports which one answered:

```
primary   groq/openai/gpt-oss-120b
              |  (call raises)
              v
fallback  groq/openai/gpt-oss-20b
```

The app shows the model that produced each answer, so the switch is visible
rather than silent.

## Choices we made, and why

**Chunk size 300.** We started at 1000, the value used in class, and it worked
badly. Page 1 of a tender is a summary table holding many facts at once, so at
1000 characters that table becomes one chunk whose embedding is an average of
many topics and matches no single question well. We tested six questions whose
answers we had checked by hand in the PDFs:

| chunk size | chunks | answers found in retrieved text |
|---|---|---|
| 300 | 441 | 5 of 6 |
| 500 | 275 | 3 of 6 |
| 800 | 171 | 3 of 6 |
| 1000 | 142 | 1 of 6 |
| 1500 | 100 | 2 of 6 |

**k = 8.** Because the chunks are small, one chunk is often only part of a
clause. On the same six questions, k=4 and k=6 both gave 5 of 6, and k=8 gave
6 of 6. Eight chunks of 300 characters is still a shorter prompt than four
chunks of 1000 would have been.

**Temperature 0.** Reading a tender is factual extraction, not creative writing.
The same question should give the same answer.

**Chroma.** It keeps the vectors, the chunk text and the metadata in one
collection and persists with a single argument. FAISS needs a separate docstore
and an index-to-id mapping, which is more moving parts than this needs.

**all-MiniLM-L6-v2 for embeddings.** Runs locally on the CPU, needs no API key
and no separate model server, and indexes all three tenders in a few seconds.
The class notebook used `nomic-embed-text` through Ollama; the LangChain
interface is identical, and this version runs with nothing but `pip install`.

## Limitations

- The embedding model does not reliably connect an abbreviation to its full
  form. "earnest money deposit" against "EMD" scores only about 0.26 cosine
  similarity, where two differently-worded full phrases ("last date for
  submission of bids" vs "closing of bids") score 0.73. Retrieval still works,
  because 0.26 is far above an unrelated sentence, but it is a real weakness.
- One of the three tenders is a scanned document whose OCR text contains errors
  ("75 davs" instead of "75 days", "t4/0e/2020" instead of a date). That noise
  makes retrieval on that document less reliable.
- Retrieval was judged with six questions we checked by hand. That is a sanity
  check, not a proper evaluation against a labelled test set, and we do not
  claim a recall figure.
- Only PDFs that already contain a text layer work. A purely scanned PDF would
  need OCR, which we did not add.
- If the right clause is not in the top 8 chunks, the system answers "this is
  not stated" rather than finding it. It fails safe, but it does fail.
- The system answers questions about clauses. It does not compare tenders,
  track amendments, or decide whether to bid.

## Possible improvements

- Hybrid retrieval (BM25 together with semantic search). This would directly fix
  the abbreviation weakness above, because BM25 matches the literal string
  "EMD", and would also catch exact tender reference numbers.
- A labelled set of questions and expected clauses, so retrieval can be measured
  properly instead of spot-checked.
- OCR for scanned tenders.

## Tech stack

LangChain, Chroma, HuggingFace sentence-transformers embeddings, Groq
(`openai/gpt-oss-120b`), LiteLLM for the fallback, Gradio for the interface.

## Team

Yuvraj Singh (27PGAI0086), Nishkarsh Khandelwal (27PGAI0081),
Darrsheni Sapovadia (27PGAI0063)
