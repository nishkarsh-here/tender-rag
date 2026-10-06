# Tender RAG

A small Retrieval Augmented Generation system that answers questions about
government tender documents, and shows which page each answer came from.

Generative AI, Assignment 3. AI & Data Science Program, Jio Institute.

**Live app:** https://tender-rag.onrender.com

It is on Render's free tier, which spins the instance down after about fifteen
minutes of no traffic, so the first request after a quiet spell takes roughly a
minute while it wakes up. After that it is quick. Open it once before you need
it rather than on the spot.

## Running it locally

Two commands, once you have a free Groq key (see [Setup](#setup)):

```bash
python -m rag.step3_embed_store   # only needed if chroma_db/ is missing
python app.py
```

Add `--share` to get a temporary public link, which is useful when
demonstrating but expires after about a week.

To host it permanently, `render.yaml` is set up for Render's free tier and
`Dockerfile` covers hosts that take a container. The app reads `PORT` when a
host sets one. The free tier allows 512 MB of memory and the app sits at about
460 MB, which is why `litellm` is imported only when the fallback is used.

## The problem

A government tender is 20 to 30 pages of dense clauses. Before bidding, a company
has to pull out a handful of facts: the earnest money deposit, the last date for
submission, the estimated cost, who the bid is addressed to, what happens if the
bid is withdrawn.

Those facts are scattered across the document and worded inconsistently. One
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
| `check_eligibility.py` | Checks a company against a tender's eligibility clauses |
| `data/companies.json` | Four example company profiles |
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
| Hybrid BM25 + semantic (measured, not used) | `rag/step4_retrieve.py` | `retrieve_hybrid()` |
| Structured output (Pydantic) | `build_tender_cards.py` | `build_card()` |
| Structured output (nested model) | `check_eligibility.py` | `check()` |
| Model fallback chain | `check_eligibility.py` | `REPORT_MODELS` |
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

## The app

The app has four tabs.

**Ask** is the tool itself.

**Can we bid?** checks a company against one tender's eligibility rules. It is
the same pipeline aimed at a different question: retrieve the clauses about who
is allowed to bid, put them beside the company profile, and ask for a structured
verdict instead of prose, so the result is a table with a page number against
every line. Four example companies are in `data/companies.json`, and you can
edit a profile in the box, upload one as `.json` or `.txt`, or upload a new
tender PDF, which is chunked and indexed on the spot.

The four companies are built so each one fits a different tender, which makes
the mismatches as informative as the matches - a Delhi display-maintenance firm
fails the Kerala lift tender on office location and electrical licence.

**Why RAG?** answers the same question twice with the same model - once with the
retrieved tender clauses in the prompt, once without - and shows both side by
side. This is the demonstration, rather than the argument, that retrieval is
doing the work.

Sometimes the model simply refuses without the context, which is harmless. The
case worth showing is when it guesses. Asked for the bid validity period of the
NITI Aayog tender with no context, it answered "typically 90 days". The document
says 75. A bidder working to 90 days misses the deadline, and nothing in the
answer warns them.

**Guided tour** takes one question and walks it through the six stages, one at a
time, with Back and Next. Each stage actually runs, so you see that question
going through this system rather than a diagram of one:

1. the raw PDF text beside the cleaned text
2. the page count turning into a chunk count, with three overlapping chunks shown
3. the question as a vector, and cosine similarity computed live on three pairs
4. the chunks that were retrieved, with their tender and page
5. the finished prompt, exactly as it is sent
6. the answer, with the model that produced it

**Tech stack** lists the six stages with the libraries each one uses and the live
numbers from the running system, the prompt template, and the measurements
behind our choices.

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

The LLM call can fail, most often because a free API tier rate-limits in the
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

**k = 10.** Because the chunks are small, one chunk is often only part of a
clause. Measured on twelve questions we checked by hand in the PDFs:

| chunks retrieved | answers found |
|---|---|
| k = 8 | 10 of 12 |
| k = 10 | 11 of 12 |
| k = 12 | 11 of 12 |
| k = 20 | 12 of 12 |

We use 10. Going to 20 does find the last one, but it doubles the amount of
text in the prompt to chase a single tender reference number.

**Hybrid retrieval: tried, measured, not kept.** Semantic search at k=8 was
missing "Who should the demand draft be drawn in favour of?", even though page 1
says "Demand Draft drawn in favour of The Director". So we built the BM25 +
semantic ensemble from the class notebook, since BM25 matches that phrase
directly.

Our first test said hybrid won, 9 of 10 against 7 of 10 - but we had written
those ten questions after watching semantic search fail, so the test was biased
towards the thing we had just built. On a fairer set of twelve questions:

| retriever | answers found |
|---|---|
| semantic only, k=8 | 10 of 12 |
| BM25 only, k=8 | 4 of 12 |
| hybrid, keep 8 | 8 of 12 |
| hybrid, keep 10 | 10 of 12 |
| semantic only, k=10 | 11 of 12 |

Hybrid fixed that one question and broke two others, because BM25 pulled in
chunks that merely repeated a keyword and pushed out the summary table holding
the answer. Raising k on plain semantic search did better and is much simpler,
so that is what we kept. `retrieve_hybrid()` is still in
`rag/step4_retrieve.py` with the numbers, so the comparison can be re-run.

**Temperature 0.** Reading a tender is factual extraction, not creative writing.
The same question should give the same answer.

**Chroma.** It keeps the vectors, the chunk text and the metadata in one
collection and persists with a single argument. FAISS needs a separate docstore
and an index-to-id mapping, which is more moving parts than this needs.

**bge-small-en-v1.5 for embeddings, through fastembed.** We began with
all-MiniLM-L6-v2 through sentence-transformers, which pulls in PyTorch at
552 MB on disk, too large for the free hosting tier. fastembed runs the model
with ONNX instead, at 76 MB. We only kept the change because it also retrieved
better: on the twelve hand-checked questions MiniLM found 11 and bge-small
found 12, including the tender reference number MiniLM never managed. The class
notebook used `nomic-embed-text` through Ollama; all three behave the same from
LangChain's side.

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

- A better fix for the abbreviation weakness. Hybrid BM25 retrieval was the
  obvious candidate and it did not work (see above), so the next thing to try is
  a stronger embedding model.
- A labelled set of questions and expected clauses, so retrieval can be measured
  properly instead of spot-checked.
- OCR for scanned tenders.

## Tech stack

LangChain, Chroma, HuggingFace sentence-transformers embeddings, Groq
(`openai/gpt-oss-120b`), LiteLLM for the fallback, Gradio for the interface.

## Team

Yuvraj Singh (27PGAI0086), Nishkarsh Khandelwal (27PGAI0081),
Darrsheni Sapovadia (27PGAI0063)
