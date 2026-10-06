# Assignment 3 Rework — Analysis Before Building

Status: analysis only. No code written yet.

Sources used:
- `Generative AI - Assignment 3.pdf` (the assignment)
- `github.com/nishkarsh-here/Generative-AI-Repo` (class notebooks 1-4, Assignment 1, Assignment 2)
- `GenAI Finals.zip` (class slides, Sessions 1-8, and notes)
- `github.com/Yuvraj0208/bidproof` (the old project, cloned and read)

---

## 1. Assignment 3: what exactly are we asked to do?

Straight from the PDF:

| Section | What it says |
|---|---|
| 1. Overview | Group of 3. "Implement the concepts learned in class in the form of a mini-project." |
| 3. Project Scope | "The project should be on **RAG** with custom datasets and problem statements." "It can **optionally** have agents." |
| 4. Idea submission | Group names/IDs, problem statement + why it is a good topic, proposed approach, tech stack (LangChain, or LangChain + LangGraph if agentic). Due 22 Sep 2026. |
| 5. Final submission | GitHub repo link + a **3-slide** deck: (1) Business impact, (2) Technical stack and GenAI solution architecture flow, (3) Appendix including AI usage disclosure. ZIP on Digiicampus by **all** members. |
| 6. Live demo | **8-minute** presentation and live demo, with 3-4 minutes of Q&A. |
| 7. Tech stack | "Use mainly the tech stack **covered in class, with syntax as per class**. If any additional syntax is used, be prepared to answer questions regarding its usage." |
| 8. Use of AI tools | AI is a helper only. "If AI-generated code is detected, significant marks (**30% flat - no discussions**) will be deducted." Brief disclosure needed in the appendix. |
| 9. Evaluation | Technical **80%**, Presentation **20%**. "**The demo will be the main grading.**" |
| Note | Late submission by any member, or editing GitHub after the deadline, can mean 0 for the whole group. |

Three lines decide everything:

1. **"The project should be on RAG."** RAG is the subject, not a feature.
2. **"The demo will be the main grading."** The repo is evidence; the 8 minutes is the exam.
3. **"Syntax as per class."** Class syntax is the safe syntax. Anything else must be defended.

### The chain

| Assignment requirement | Learning objective | Concept taught | Implementation | Expected demonstration |
|---|---|---|---|---|
| "on RAG" | Can you build an indexing + retrieval + generation loop? | Notebook 4, whole notebook | `step1`-`step6` files | Run the pipeline live, stage by stage |
| "custom dataset" | Can you take a messy real document and make it searchable? | Document loaders, regex cleaning | Real tender PDFs in `data/` | Show the PDF, show the extracted text |
| "optionally agents" | Do you know when NOT to add something? | LangGraph, Assignment 2 Part 2 | At most a 2-node graph | Explain why the project does not need 14 agents |
| "syntax as per class" | Did you learn, or did you import? | Notebooks 1-4 | LangChain + Groq + Ollama + Chroma | Answer any "why this line" question |
| "3-slide deck" | Can you compress? | - | 3 slides, no more | Business impact, architecture flow, appendix |
| "8-min demo" | Can you explain your own system? | - | - | Live run plus Q&A |
| "AI disclosure" | Honesty | - | Appendix slide | State what AI helped with |

### What is likely worth marks
Working end-to-end RAG. Clearly located pipeline steps. Correct use of class libraries. Sensible, defensible hyperparameters. Grounded answers with sources. Correct behaviour when the answer is not in the document. A real business problem. A clean 8-minute demo.

### What is optional
Gradio UI. Structured Pydantic output. LangGraph. Hybrid BM25 retrieval. Chunk-size comparison. Multi-document metadata filtering.

### What is unnecessary
Deployment, hosting, Docker, databases and migrations, authentication, multi-tenancy, REST APIs, React frontends, OCR engines, portal scrapers, tracing and observability, large test suites, agent orchestration.

### What counts as overengineering here
Anything that puts a layer between the reader and the eight RAG steps. Every abstraction costs demo time and adds a question we may not be able to answer in 3 minutes.

### What a teacher expects a student to explain in a viva
Why chunk at all. Why this chunk size. What an embedding is. Why cosine similarity. What is stored in the vector store. What top-k means. What exactly goes into the prompt. Why this is different from asking ChatGPT. What happens when the answer is not in the document.

### What makes a teacher feel we genuinely understood
Being able to point at one line and say what it does and why we chose it - without opening three other files first.

---

## 2. What is the teacher actually trying to teach?

From the four notebooks, the teaching is sequential and deliberate:

- **Notebook 1 - LLM Generation Parameters.** An LLM is a sampler you control. temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop, seed. Each one is demonstrated three times at three settings so the student *sees* the difference.
- **Notebook 2 - Introduction to LangChain.** What LangChain actually buys you. He deliberately writes the same task **twice** - once in raw OpenAI client syntax, once in LangChain - so you understand the abstraction instead of trusting it. Then output parsers, `ChatPromptTemplate`, and LCEL (`template | llm | StrOutputParser()`).
- **Notebook 3 - Structured Output Generation.** Four different ways to get JSON out, from worst to best: plain prompting, native `response_format`, `JsonOutputParser`, `JsonOutputParser` + Pydantic, and `with_structured_output`.
- **Notebook 4 - RAG.** The core. Part 1 Indexing: tokenization (tiktoken, AutoTokenizer), embeddings (`OllamaEmbeddings`, `nomic-embed-text`), cosine similarity written out by hand in NumPy, document loading (`WebBaseLoader`, `UnstructuredPDFLoader`), regex cleaning, `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)`, `SemanticChunker`, FAISS and Chroma. Part 2 Retrieval and Generation: `as_retriever`, score threshold, top-k, metadata filter, MMR, hybrid BM25 + `EnsembleRetriever`, `init_chat_model` on Groq, a custom `ChatPromptTemplate`, the hub prompt `rlm/rag-prompt`, then the full LCEL chain with `RunnablePassthrough`.

The lesson is: **you should be able to see each stage of the pipeline and inspect what came out of it.**

---

## 3. Which class concepts matter for this assignment?

Ranked by how likely they are to be asked about.

**Must be visible in our project**
- Document loading (Notebook 4)
- Text cleaning with regex (Notebook 4, cell 57)
- `RecursiveCharacterTextSplitter` with chunk size and overlap (Notebook 4)
- `OllamaEmbeddings` with `nomic-embed-text` (Notebook 4, and Assignment 2)
- Cosine similarity, conceptually (Notebook 4)
- A vector store, Chroma or FAISS (Notebook 4)
- `as_retriever` with `search_kwargs={"k": ...}` (Notebook 4)
- `ChatPromptTemplate` with a `{context}` and `{question}` slot (Notebook 4)
- `init_chat_model(model, model_provider="groq")` (Notebooks 2, 3, 4)
- LCEL chain `prompt | llm | StrOutputParser()` (Notebooks 2, 4)
- The out-of-context question and the "say you don't know" instruction (Notebook 4, cells 155-161)
- temperature chosen on purpose (Notebook 1)

**Worth adding if cheap**
- Pydantic + `PydanticOutputParser` or `with_structured_output` (Notebook 3, Assignment 2)
- `OutputParserException` handling (Assignment 2)
- Gradio Blocks UI (Assignment 2)
- Metadata filtering on retrieval (Notebook 4)
- tiktoken token counting (Notebook 4)
- A 2-node LangGraph: retrieve -> generate (Assignment 2 Part 2)

**Taught but not needed here**
- `SemanticChunker` (slow, and the class notes say so)
- MMR, hybrid BM25 + RRF (nice, but only if there is time)
- AutoTokenizer / HuggingFace gated models
- `frequency_penalty`, `presence_penalty`, `seed`, `stop`

---

## 4. What does the teacher appear to value?

Read off his own code, not guessed:

1. **Inspect everything.** After nearly every step he prints the result. `print(f"Total characters: ...")`, loops that print every chunk, `np.array(embeddings).shape`, `type(response)` over and over. He wants intermediate state visible.
2. **Do it the long way first, then the shortcut.** Raw OpenAI client before LangChain. Manual three-step `template.invoke` -> `llm.invoke` -> `parser.invoke` before showing `template | llm | parser`. Manual RAG test before the LCEL chain.
3. **Flat, procedural code.** In four notebooks and a full assignment solution there is not one custom class except Pydantic schemas and a `TypedDict` state. Functions and top-level statements only.
4. **Hyperparameters are choices.** He writes `# hyperparameter that can be tuned` in the margin, three separate times.
5. **Open and local where possible.** Ollama, `nomic-embed-text`, Groq free tier, Chroma. No paid infrastructure.
6. **Honesty about limits.** He highlights that the hub RAG prompt says "say I don't know", and calls that a best practice. He runs an out-of-context question on purpose.
7. **Questions in the margins.** "What does this do?" "What is the min and max of cosine similarity?" "Can we use an embedding model with a different dimension from the chat model?" He expects students to be able to answer these.

### What he does not expect students to build
Services, APIs, frontends, databases, containers, deployments, orchestration layers. None of it appears anywhere in the course material. Assignment 2, his own full solution, is **one notebook plus a CSV**.

---

## 5. What did BidProof do well?

Credit where it is due, and the teacher said most of this himself:

- **The problem is genuinely good.** Indian public tenders are long, dense, and a missed clause means disqualification. This is a real industry problem with real money attached.
- **The pitch and narrative were strong.** The deck and the "proof chain" framing worked.
- **It actually ran.** Deployed and demo-able at a public URL.
- **The engineering is honest.** This is the part worth preserving. The code says out loud when something is not real: `evaluation/pipeline.py:251` literally records that retrieval is not implemented and that reporting a score "would be measuring something that does not exist." That intellectual honesty is rare and should carry into the new project.
- **The model-routing idea was good.** Naming roles (`small` / `mid` / `strong`) instead of hardcoding vendors is a sound idea, and it is what the teacher liked.
- **Graceful degradation existed.** When the model was unavailable or returned something ungrounded, `chat.py` fell back to quoting the matched clauses with their page numbers rather than inventing prose.

---

## 6. What did BidProof do badly?

This is the uncomfortable part, and it needs to be said plainly.

**BidProof had no RAG in it.**

Not "hard-to-find RAG". None. Verified by reading the repository:

| What we should have found | What is actually there |
|---|---|
| Embeddings | **Zero.** No embedding model anywhere in application code. `evaluation/pipeline.py:251`: "There is no embedding retrieval in the product yet." |
| Vector store | **Zero.** `infra/deploy/bootstrap_db.py:43` creates the `pgvector` extension, and nothing ever uses it. |
| Chunking | **Zero** for retrieval. The only hit for "chunk" is HTML parsing in a GeM portal adapter. |
| Semantic retrieval | **Zero.** `agents/matcher/bidproof_matcher/retrieval.py` is 48 lines of keyword set-overlap with a stopword list. `chat.py` scores by token overlap too. |
| LangChain | **Zero imports in application code.** LangChain appears only in `main.py` to *disable* its tracer. |
| Class-style document loading | **Zero.** `agents/parser/.../ladder.py` plus five engines (pypdfium2, Docling, PaddleOCR-VL, RapidOCR) - about 721 lines, none of it from the class stack. |

So when the teacher said "we were unable to pinpoint the RAG specific lines in the code", the honest answer is: **there were none to pinpoint.**

Scale, for context:

| Measure | BidProof |
|---|---|
| Python files | 225 |
| Lines of Python | ~26,945 |
| React `.tsx` files | 55 |
| Agent packages | 14 |
| Test files | 62 |
| README | 444 lines |
| `docs/FINISH_STATUS.md` | 1,293 lines |
| Infrastructure | Docker, Postgres, Alembic migrations, LiteLLM proxy container, Langfuse, Render hosting |

Also: the LiteLLM part was not actually a fallback. `infra/litellm/config.yaml` runs a **separate proxy container** mapping three roles to env vars, and `num_retries: 2` is a retry, not a model-to-model fallback. The real fallback was "model fails -> deterministic template", in `chat.py`. The teacher liked the *idea*. The implementation was a container, a config file, an 86-line gateway and a 178-line availability checker - about 264 lines plus infrastructure to express something that should be twelve lines.

---

## 7. Why was the teacher disappointed?

Not because the work was weak. Because it answered a different question.

The assignment said: *build a RAG project, agents optional.*
We built: *an agentic platform, with no RAG.*

Then three things compounded it:

1. **He could not verify learning.** Grading is 80% technical and the demo is the main grading. If he cannot find the concepts from his own notebooks in the code, he has nothing to award marks against. 26,945 lines of code he cannot map to his syllabus is worth less than 150 lines he can.
2. **The tech stack was off-syllabus.** Section 7 asks for class syntax. We used FastAPI, SQLAlchemy, Alembic, httpx, Docling, PaddleOCR, React, Docker. He taught LangChain, Ollama, Chroma, Groq. Almost nothing lined up.
3. **It triggers the AI-code suspicion in Section 8.** A 225-file, 27,000-line, 14-agent system from a 3-person student group in a few weeks invites exactly the question that carries a flat 30% penalty. "Too complete" was not a compliment - it was a warning.

"Looked like a prototype suitable for a hackathon" is the key phrase. Hackathon projects are judged on impression. This is judged on demonstrated understanding.

---

## 8. Where exactly did the architecture become too complex?

Ordered by damage done:

1. **14 agents as separate installable packages.** `scout`, `triage`, `parser`, `extractor`, `matcher`, `factchecker`, `riskscorer`, `decider`, `librarian`, `proposalwriter`, `questionwriter`, `formfiller`, `guard`, `amendmentwatcher`, plus a `conductor`. The assignment said agents were optional. Each one adds a package boundary to trace through.
2. **The LangGraph conductor.** Orchestration on top of 14 agents. To follow one request you read the graph, then the node, then the service, then the agent package, then the engine.
3. **A full web application.** FastAPI backend plus a React frontend with 55 components. None of this is GenAI. All of it is reading time.
4. **Persistence and migrations.** Postgres, SQLAlchemy models, Alembic versions, seed scripts, a bootstrap script. For a mini-project, a local Chroma folder does the whole job.
5. **The LiteLLM proxy as infrastructure.** A Docker container, a role config, a gateway client, an availability checker. The good idea got buried in its own plumbing.
6. **The parser ladder.** Four extraction engines with a routing ladder and a ground-check, ~721 lines, to do what `PyPDFLoader` does in one line for our purpose.
7. **Portal adapters.** CPPP, GeM, HTML portal, NICEPROC scrapers. Entirely outside the assignment.
8. **62 test files and a 1,293-line status document.** The documentation itself became something nobody could read in one sitting.

The pattern: **every layer was individually defensible and collectively fatal.** Each one pushed the actual GenAI content further from the surface, until there was nothing visible at the surface at all.

---

## 9. What should we keep?

| Keep | Why |
|---|---|
| The tender / bid problem domain | It is genuinely good, the teacher said so, and our idea submission already locked this problem statement |
| The business framing from Slide 1 | It worked |
| LiteLLM fallback, as a concept | He explicitly liked it |
| Grounded answers with citations | Quoting the clause and page is the right behaviour for this domain, and it is what RAG is for |
| Honesty about what is and is not implemented | The best habit in the old repo |
| Real tender PDFs | A genuine custom dataset, which Section 3 requires |

## 10. What should we remove?

Everything else. Specifically: all 14 agents, the conductor, FastAPI, React, Postgres, Alembic, Docker, the deployment, the portal adapters, the OCR ladder, the LiteLLM proxy container, the gateway and availability modules, the 62 test files, the long spec documents, the auth and multi-tenancy, and Langfuse tracing.

Not because they are bad engineering. Because none of them demonstrate a single thing taught in this course.

---

## 11. What should the new project demonstrate?

1. A complete RAG pipeline, with every stage separately visible and inspectable.
2. That we understand *why* each stage exists, not just that it runs.
3. That our hyperparameters were chosen, not copied.
4. That the system knows when it does not know.
5. That answers are traceable to a specific chunk and page of a specific tender.
6. That we can swap models and survive a model failure.
7. That the problem is real and the solution fits it.

## 12. What should the new project deliberately NOT demonstrate?

Scale. Completeness. Deployment. Production readiness. Architectural sophistication. Breadth of tooling. Agent orchestration. Test coverage.

It should be obvious at a glance that this is a **learning project**, built by three students who understand every line.

---

## 13. How will the new architecture be different?

Old: a platform with a GenAI feature buried in it, where the pipeline is the thing you cannot see.
New: a pipeline, with a thin UI on top, where the pipeline is the only thing you can see.

Proposed flow:

```
Tender PDF (data/tenders/*.pdf)
        |
        v  step1_load.py      load_tender_pdf()   -> raw text + page numbers
        |                     clean_text()        -> tidy text
        v  step2_chunk.py     split_into_chunks() -> ~N chunks with metadata
        |
        v  step3_embed_store.py  create_embeddings() -> vectors
        |                        store_chunks()      -> Chroma on disk
        |
   [ user question ]
        |
        v  step4_retrieve.py  retrieve_chunks()   -> top-k chunks + scores
        |
        v  step5_prompt.py    build_prompt()      -> context + question
        |
        v  step6_generate.py  generate_answer()        -> LCEL, class syntax
        |                     answer_with_fallback()   -> LiteLLM primary/fallback
        v
   Answer + the exact source chunks and page numbers
```

Six files. One function per concept. The filenames are the pipeline.

---

## 14. How will we make the RAG impossible to miss?

Four moves, in order of effect:

1. **Number the files after the pipeline stages.** `step1_load.py` through `step6_generate.py`. The teacher opens the repo and the directory listing *is* the RAG pipeline. "Show me your RAG" is answered before anyone speaks.
2. **One concept per function, named after the concept.** `load_tender_pdf`, `clean_text`, `split_into_chunks`, `create_embeddings`, `store_chunks`, `retrieve_chunks`, `build_prompt`, `generate_answer`. No indirection, no registries, no base classes.
3. **A "Where each concept lives" table at the top of the README.** Concept -> file -> function -> line. One row per class concept.
4. **A notebook that runs the pipeline stage by stage and prints the output of each stage** - exactly the teacher's own habit. Show the raw text. Show chunk counts. Show one chunk. Show the embedding shape. Show the retrieved chunks with scores *before* the LLM is called. Show the final prompt string. Then show the answer.

That last point matters most. If he can see the assembled prompt with the retrieved context inside it, there is no remaining question about whether RAG is happening.

---

## 15. How will LiteLLM fallback be demonstrated?

One function, in `step6_generate.py`, roughly:

```python
PRIMARY_MODEL  = "groq/openai/gpt-oss-120b"
FALLBACK_MODEL = "ollama/llama3.2"

def answer_with_fallback(prompt_text):
    """Try the primary model. If it fails, use the fallback model."""
    try:
        response = litellm.completion(model=PRIMARY_MODEL,
                                      messages=[{"role": "user", "content": prompt_text}])
        return response.choices[0].message.content, PRIMARY_MODEL
    except Exception as error:
        print("Primary model failed:", error, "- switching to", FALLBACK_MODEL)
        response = litellm.completion(model=FALLBACK_MODEL,
                                      messages=[{"role": "user", "content": prompt_text}])
        return response.choices[0].message.content, FALLBACK_MODEL
```

About twelve lines. The `litellm` Python library, not a proxy, not a container, not a gateway class.

**Demo:** ask a question and show it answered by Groq. Then break the key (`GROQ_API_KEY=wrong`) and ask again. The console prints the failure, the local Ollama model answers, and the UI shows which model produced the answer.

**One caution.** LiteLLM is *not* in the class material - it is not in the course `pyproject.toml`. Section 7 allows additional syntax but says be ready to answer questions about it. So we keep the main generation path in pure class syntax (`init_chat_model` + LCEL) and present LiteLLM as a clearly-labelled resilience layer beside it. Both functions live in the same short file. That way the "as per class" requirement and the teacher's own compliment are both satisfied, and neither hides the other.

---

## 16. The Minimum Viable GenAI Project

### CORE - build this first, and it is already a complete submission

| # | Item | Class source |
|---|---|---|
| 1 | 2-3 real public tender PDFs in `data/tenders/` | Section 3, "custom datasets" |
| 2 | `load_tender_pdf()` + `clean_text()` | Notebook 4, loaders + regex cleaning |
| 3 | `split_into_chunks()` with `RecursiveCharacterTextSplitter` | Notebook 4 |
| 4 | `create_embeddings()` with `OllamaEmbeddings("nomic-embed-text")` | Notebook 4, Assignment 2 |
| 5 | `store_chunks()` into Chroma with a persist directory | Notebook 4, Assignment 2 |
| 6 | `retrieve_chunks()` via `as_retriever(search_kwargs={"k": 4})` | Notebook 4 |
| 7 | `build_prompt()` with `ChatPromptTemplate`, `{context}` + `{question}` | Notebook 4 |
| 8 | `generate_answer()` with `init_chat_model` on Groq + LCEL | Notebooks 2, 4 |
| 9 | Answer displayed with its source chunks and page numbers | Notebook 4 |
| 10 | "Not in this tender" behaviour, plus an out-of-context test question | Notebook 4, cells 155-161 |
| 11 | `answer_with_fallback()` via LiteLLM | Teacher feedback |
| 12 | Teaching notebook that prints the output of every stage | Teacher's own style |
| 13 | README with the concept-to-file table | Teacher feedback |
| 14 | `TEACHER_WALKTHROUGH.md` | Our own insurance for the viva |
| 15 | 3-slide deck with the AI disclosure | Section 5 |

### OPTIONAL - add only if Core is finished and demo time allows

| Item | Cost | What it earns |
|---|---|---|
| Gradio Blocks UI | Low - Assignment 2 has the pattern | Makes the 8-minute demo much smoother |
| Pydantic structured answer (answer, clause, page, found_in_document) | Low | Directly demonstrates Notebook 3 |
| Chunk-size comparison, 500 vs 1000 vs 1500, honestly measured | Low | Proves hyperparameters were chosen |
| Metadata filter to query one tender out of several | Low | Demonstrates Notebook 4 metadata filtering |
| 2-node LangGraph: retrieve -> generate, with the mermaid diagram | Medium | Mirrors Assignment 2 Part 2, covers "optionally agentic" |
| Hybrid BM25 + semantic `EnsembleRetriever` | Medium | Demonstrates advanced retrieval |

### AVOID - decided in advance, no re-litigating mid-build

Agents and orchestration. FastAPI or any backend service. React or any frontend build. Postgres, SQLAlchemy, Alembic, pgvector. Docker. Deployment and hosting. The LiteLLM proxy, gateway, roles, availability checks. Portal scrapers and adapters. OCR engines and parser ladders. Auth, orgs, multi-tenancy. Tracing and observability. Large test suites. Config frameworks. Custom classes, factories, registries, protocols, abstract interfaces. Long specification documents.

---

## 17. Proposed repository structure

```
tender-rag/
├── README.md                      # project + the concept-to-file table
├── TEACHER_WALKTHROUGH.md         # question -> answer -> file -> function -> what to show
├── requirements.txt
├── .env.example                   # GROQ_API_KEY
├── data/
│   └── tenders/                   # 2-3 real public tender PDFs
├── rag/
│   ├── step1_load.py              # load_tender_pdf(), clean_text()
│   ├── step2_chunk.py             # split_into_chunks()
│   ├── step3_embed_store.py       # create_embeddings(), store_chunks(), load_store()
│   ├── step4_retrieve.py          # retrieve_chunks()
│   ├── step5_prompt.py            # build_prompt()  - the RAG prompt template
│   └── step6_generate.py          # generate_answer(), answer_with_fallback()
├── notebooks/
│   └── 01_rag_pipeline_explained.ipynb    # the teaching notebook
└── app.py                         # Gradio UI, ~80 lines, Assignment 2 style
```

Target size: **6 pipeline files, roughly 40-70 lines each. Under 500 lines of Python in total.** If it grows past that, something has crept in that should not have.

Compare: BidProof was 225 files and 26,945 lines.

---

## 18. Class concept -> implementation mapping

This table is the checklist. If a row cannot be filled honestly, the row gets deleted - we do not add technology to fill a row.

| Class concept | Why we use it | Where it appears | File | Function | How we explain it |
|---|---|---|---|---|---|
| Document loading | A tender is a PDF; we need its text | Indexing | `step1_load.py` | `load_tender_pdf()` | "LangChain's PDF loader gives one Document per page, so we keep page numbers for citations" |
| Text cleaning | Tender PDFs come out with broken spacing and repeated headers | Indexing | `step1_load.py` | `clean_text()` | "Regex collapses blank lines and repeated spaces - same cleaning step as the class notebook" |
| Chunking | Documents are too long for the context window, and smaller pieces retrieve more precisely | Indexing | `step2_chunk.py` | `split_into_chunks()` | "`RecursiveCharacterTextSplitter`, size 1000, overlap 200, so a clause split across a boundary still survives in one chunk" |
| Embeddings | Text must become numbers before meaning can be compared | Indexing | `step3_embed_store.py` | `create_embeddings()` | "`nomic-embed-text` on Ollama, same model as class, runs locally and free" |
| Cosine similarity | How closeness between question and chunk is measured | Retrieval | `step4_retrieve.py` | - (Chroma default) | "Chroma does it internally; the class notebook wrote it out in NumPy, and we can reproduce that" |
| Vector store | Somewhere to keep vectors plus text plus metadata and search them fast | Indexing | `step3_embed_store.py` | `store_chunks()` | "Chroma with `persist_directory`, so indexing runs once and the demo starts instantly" |
| Retrieval / top-k | Only the relevant chunks should reach the model | Retrieval | `step4_retrieve.py` | `retrieve_chunks()` | "`as_retriever(search_kwargs={'k': 4})` - four chunks is enough context without drowning the prompt" |
| Metadata | Citations and per-tender filtering | Indexing + retrieval | `step2_chunk.py`, `step4_retrieve.py` | - | "Each chunk carries source file and page, which is what makes the citation possible" |
| Prompt template | The retrieved context has to be placed into the question | Generation | `step5_prompt.py` | `build_prompt()` | "`ChatPromptTemplate` with `{context}` and `{question}`, plus the instruction to say so when the tender does not cover it" |
| Chat model | The generator | Generation | `step6_generate.py` | `generate_answer()` | "`init_chat_model('openai/gpt-oss-120b', model_provider='groq')` - same model as class" |
| Generation parameters | Factual extraction should not be creative | Generation | `step6_generate.py` | `generate_answer()` | "temperature 0.0-0.2; Notebook 1's own table puts deterministic extraction tasks at 0.2" |
| LCEL chain | The class's recommended way to connect the pieces | Generation | `step6_generate.py` | `generate_answer()` | "`prompt \| llm \| StrOutputParser()` - exactly the notebook's final RAG chain" |
| Grounding / "I don't know" | A wrong EMD figure is worse than no answer | Generation | `step5_prompt.py` | `build_prompt()` | "The prompt forbids answering from outside the context; we demo it with an out-of-context question" |
| LiteLLM fallback | Free-tier Groq can rate-limit mid-demo | Generation | `step6_generate.py` | `answer_with_fallback()` | "try primary, catch the exception, re-send the same prompt to a local model" |
| *(optional)* Structured output | Answer plus clause plus page as fields, not prose | Generation | `step6_generate.py` | - | "Pydantic schema, same as Notebook 3 and Assignment 2" |
| *(optional)* LangGraph | The assignment allows an agentic variant | Generation | notebook | `retrieve` / `generate` nodes | "Two nodes and three edges, same shape as Assignment 2 Part 2" |

---

## 19. Old vs new architecture

| Area | Old BidProof | The problem | New approach |
|---|---|---|---|
| Architecture | 14 agent packages + LangGraph conductor + API + web app | The GenAI content is invisible under the platform | 6 numbered pipeline files + 1 Gradio file |
| Files | 225 Python, 55 TSX, ~27,000 lines | Nobody in the group could hold it in their head | ~8 Python files, under 500 lines |
| Agents | 14 | Section 3 says agents are *optional*; RAG was mandatory and missing | 0, or one optional 2-node graph |
| Document processing | 4-engine OCR ladder, Docling, PaddleOCR, ~721 lines | Off-syllabus, huge, unexplainable in a viva | One LangChain PDF loader, ~15 lines |
| Chunking | None | The assignment's core concept, absent | `RecursiveCharacterTextSplitter`, visible, tuned |
| Embeddings | None | The assignment's core concept, absent | `OllamaEmbeddings("nomic-embed-text")` |
| Vector store | pgvector extension created, never used | Looked like RAG in the infra, was not | Chroma, persisted to a local folder |
| Retrieval | 48-line keyword overlap | Not semantic retrieval; not RAG | Semantic top-k via `as_retriever` |
| Prompting | Prompts scattered across agent packages | Hard to find, hard to point at | One prompt, in `step5_prompt.py` |
| LLM access | Role-based gateway over an HTTP proxy | Three layers between code and model | `init_chat_model`, direct, class syntax |
| LiteLLM | Docker container + config + 264 lines of client code | The good idea, buried | One 12-line function |
| Fallback | "model fails -> deterministic template", inside `chat.py` | Real, but not a model-to-model fallback | Primary model -> fallback model, demonstrable live |
| Testing | 62 test files | Signals production intent, costs demo time | A few sanity checks in the notebook |
| Deployment | Docker, Postgres, Render, Alembic | Zero marks, large risk, lots of time | None - runs locally |
| Documentation | 444-line README, 1,293-line status doc | Too long to read, reads like a product | Short README + walkthrough built for viva questions |
| Complexity | Very high | Triggered the "too complete / AI-generated" concern | Deliberately low |
| Explainability | Low - the RAG lines did not exist | The single reason marks were lost | Every concept has one named function |

---

## 20. How this recovers the lost marks

| Teacher's comment | Why it cost marks | New design response | Evidence in the repo |
|---|---|---|---|
| "Build a simple RAG based GenAI project" | There was no RAG. Section 3 made RAG the required subject. | A real RAG pipeline is the entire project | `rag/step1_...step6_`, and the notebook running them in order |
| "Proper and simple code files" | 225 files, 14 packages, no file was self-explanatory | Six files named after the six stages, one concept per function | The directory listing |
| "Step by step guide and documentation" | 444-line README that read like a product page | README with setup, run steps, and a concept-to-file table | `README.md` |
| "Proper showcase of what is being taught in class" | Almost no class library appeared anywhere | LangChain, Ollama embeddings, Chroma, Groq, LCEL, ChatPromptTemplate - all present | Section 18 mapping table |
| "Unable to pinpoint the RAG specific lines" | The lines genuinely did not exist | Numbered files, named functions, printed intermediate output | `step1`-`step6`, plus the notebook showing each stage's output |
| "LiteLLM fallback was good" | Good idea, but buried under a proxy container | Kept, reduced to twelve visible lines, demoed live by breaking the key | `step6_generate.py::answer_with_fallback` |
| "Good pitch / good industry use case" | Not a loss - keep it | Same domain, same business framing | Slide 1 and the README overview |
| "Too complex and too complete" | Invited the Section 8 AI-code suspicion and left nothing gradeable | Under 500 lines, every line explainable by any of the three of us | The whole repo |
| "Looked like a hackathon prototype" | Judged on impression, not understanding | Looks like a learning project, because it is one | Notebook structure and README tone |

---

## 21. Evaluating the new design as the teacher would

| Question | Verdict |
|---|---|
| Can I immediately find the RAG code? | Yes. The filenames are the pipeline. |
| Can the student explain chunking? | Yes - one function, and the notebook prints chunk counts and a sample chunk. |
| Can the student explain embeddings? | Yes - same model as class, and the notebook prints the vector shape. |
| Can the student explain retrieval? | Yes - retrieved chunks are printed with scores before the LLM is called. |
| Can the student explain the prompt? | Yes - the notebook prints the fully assembled prompt string. |
| Can the student explain the LLM call? | Yes - `init_chat_model` plus an LCEL chain, straight from the notebook. |
| Can the student explain LiteLLM? | Yes - twelve lines, in one file. |
| Can the student explain the fallback? | Yes, and demonstrate it live by breaking the API key. |
| Does it reflect the class material? | Yes - the mapping table is explicit about which notebook each piece comes from. |
| Is it unnecessarily complicated? | No. |
| Can the student understand every important file? | Yes - that is the design constraint. |
| Does it look like an academic learning project? | Yes. |
| Does it still have a meaningful industry use case? | Yes - the tender problem is unchanged. |

**Remaining risks a teacher might still raise**
- "This is quite small." Answer: Section 3 asked for RAG on a custom dataset; this does that completely, and the optional extras (structured output, LangGraph, hybrid retrieval) are there to show depth without bulk.
- "Why LiteLLM when class taught `init_chat_model`?" Answer: both are present, side by side, and we can explain why we would use each.
- "Did you actually test this?" We must only claim what we actually ran. No invented numbers anywhere.
- "You had a big project before - what happened?" Answer honestly: we built the wrong thing, and this is what the assignment asked for.

---

## 22. Likely viva questions

Format: **Question -> Answer -> File -> Function -> What to show.**

**On chunking**
- *Where are you chunking?* -> `rag/step2_chunk.py`, `split_into_chunks()`. Show the file, then the notebook cell printing `len(chunks)`.
- *Why do you need chunking?* -> A tender runs to 100+ pages, far past the context window, and sending the whole document buries the relevant clause in noise. Smaller pieces retrieve more precisely.
- *Why this chunk size?* -> 1000 characters with 200 overlap, the class default, kept after checking 500 and 1500. Tender clauses are short paragraphs; 1000 usually holds a whole clause. The overlap means a clause straddling a boundary still appears intact in one chunk. (Only say "we checked" if we actually did.)
- *What does `add_start_index` do?* -> Records where the chunk began in the original text, which helps trace an answer back to the source.

**On embeddings**
- *Where are embeddings generated?* -> `step3_embed_store.py`, `create_embeddings()`.
- *Why are embeddings required?* -> Keyword matching misses "EMD" when the tender says "earnest money deposit". Embeddings place similar meanings near each other in vector space, so we can search by meaning.
- *Why this embedding model?* -> `nomic-embed-text` on Ollama - the model used in class, runs locally, free, no API limits during a demo.
- *How are question and chunks compared?* -> Cosine similarity. Chroma does it internally; the class notebook wrote it out in NumPy and we can reproduce that.
- *Can the embedding model have a different dimension from the chat model?* -> Yes, they are unrelated. The embedding dimension only has to match between indexing and querying. (This is the teacher's own margin question in Notebook 4.)

**On the vector store**
- *Where are embeddings stored?* -> `step3_embed_store.py`, `store_chunks()`, into Chroma with a persist directory.
- *Why Chroma and not FAISS?* -> Class covered both. Chroma keeps vectors, text and metadata in one collection and persists with one argument. FAISS needs a separate docstore and an index-to-id map. Chroma is simpler for this size.

**On retrieval**
- *Where is retrieval happening?* -> `step4_retrieve.py`, `retrieve_chunks()`.
- *What exactly is retrieved?* -> LangChain `Document` objects - `page_content` plus metadata with source file and page number.
- *Why k=4?* -> Enough context to cover a clause and its neighbours, small enough to keep the prompt tight and the answer focused.

**On the prompt**
- *Where is the prompt built?* -> `step5_prompt.py`, `build_prompt()`.
- *Where is retrieved context inserted?* -> The `{context}` placeholder in the `ChatPromptTemplate`. Show the printed, fully assembled prompt in the notebook - this is the single most convincing thing to show.
- *How is this different from asking ChatGPT?* -> ChatGPT has never seen this tender. It would either refuse or invent an EMD amount. We retrieve the actual clause from this specific PDF and the answer cites the page it came from.
- *What if the answer is not in the document?* -> The prompt instructs the model to say the tender does not cover it. Demo with a deliberately out-of-scope question, the same way the class notebook does.

**On the LLM and LiteLLM**
- *Where is the LLM called?* -> `step6_generate.py`, `generate_answer()`, via `init_chat_model` and an LCEL chain.
- *What temperature, and why?* -> Low. This is factual extraction, not creative writing; Notebook 1's own table puts deterministic tasks around 0.2.
- *Where is LiteLLM used?* -> `step6_generate.py`, `answer_with_fallback()`.
- *What happens when the first model fails?* -> The exception is caught, the same prompt goes to the local fallback model, and the UI reports which model answered.
- *Why LiteLLM when we taught `init_chat_model`?* -> Both are in the file. LangChain is the class path; LiteLLM gives one uniform call across providers and makes the fallback a four-line `try/except`, which matters on a free tier during a live demo.

**On the project overall**
- *Which class concepts are demonstrated?* -> Point at the README mapping table.
- *Why no agents?* -> Section 3 makes agents optional and RAG required. We spent the effort on RAG. *(If the optional LangGraph is built: and here is a two-node graph showing we can do it.)*
- *What are the limitations?* -> Only answers what is in the indexed tenders; scanned PDFs without a text layer would need OCR; retrieval quality was judged by reading results, not a labelled test set.
- *What did AI help with?* -> Answer exactly and honestly, matching the appendix slide.

---

## 23. Risks that could still make this too complex

1. **Adding every optional item.** Gradio + Pydantic + LangGraph + hybrid retrieval + multi-tender filtering together rebuilds the old problem in miniature. Pick at most two or three, after Core works.
2. **`app.py` growing.** If the Gradio file passes about 100 lines, stop. Assignment 2's UI is the ceiling.
3. **The notebook turning into a benchmark suite.** A short honest chunk-size comparison is good. An evaluation framework is not.
4. **Inventing results.** Section 8 and basic honesty both forbid it. Only report numbers we actually produced. If we did not test retrieval quality formally, say so in Limitations.
5. **Importing BidProof's parser "because it is better".** It is better, and it is also 721 off-syllabus lines. One LangChain loader.
6. **LiteLLM creeping back into a gateway.** One function. If a `class` appears near it, revert.
7. **README drifting back to product-marketing tone.** The audience is one teacher checking whether we learned RAG.
8. **Building for an imagined "wow".** The grading is 80% technical and the demo is the main event. Clarity is the wow.
9. **Splitting work so that only one person understands each file.** All three of us must be able to answer any question - Q&A is 3-4 minutes and he can ask anyone.
10. **Editing the repo after the deadline.** The assignment note says this can mean zero. Freeze it.

---

## 24. Final strategy

1. Keep the tender and bid domain. It is good, and our idea submission already locked it.
2. Treat this as building the project the assignment asked for, not as shrinking BidProof. Start from an empty folder.
3. Build Core in order, step1 through step6, testing each stage before the next.
4. Use class syntax everywhere. Deviate only for LiteLLM, and keep the class path beside it.
5. Make every stage print its output in the notebook. That is the teacher's own habit and it is what makes learning visible.
6. Write the README and the Teacher Walkthrough *while* building, not after - they are what converts working code into marks.
7. Add optional items only after Core runs end to end, and only those we can explain.
8. Rehearse the 8 minutes: problem (1 min), architecture (1 min), live run (4 min), fallback demo (1 min), limitations and learning (1 min). Then all three of us rehearse Q&A.
9. Be honest in the appendix about AI usage, and honest in the README about limitations.
10. Freeze the repo before the deadline.

**This is the project we should build and this is why it should satisfy the assignment better than BidProof.**

A small tender-question-answering tool, built as six numbered files that are literally named after the six stages of a RAG pipeline, using the exact libraries from class, with a twelve-line LiteLLM fallback, a notebook that prints what comes out of every stage, and a README that maps each class concept to the function that implements it.

It is better than BidProof for this assignment for one reason that outweighs all the others: **Assignment 3 asked for a RAG project, and BidProof did not contain RAG.** It had no embeddings, no vector store, no chunking and no semantic retrieval - its own evaluation code says so. It was an impressive answer to a question nobody asked.

The new project does the opposite. It is smaller in every measurable way and larger in the only way being graded: a teacher can open it, find every concept from his own notebooks in under a minute, ask any of us about any line, and get a real answer. The business problem stays exactly as good as it was. The engineering gets out of the way of the learning.
