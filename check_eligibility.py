"""Can this company bid for this tender?

This reuses the same RAG pipeline. The only new idea is what we retrieve and
what we ask for:

  1. retrieve the clauses about who is allowed to bid   (rag/step4_retrieve.py)
  2. put those clauses and the company profile in one prompt
  3. ask for a structured verdict instead of prose

Step 3 is the structured output method from class: a Pydantic model passed to
with_structured_output, so the model returns fields we can render as a table
rather than a paragraph we would have to parse.

The answer is only as good as the clauses retrieved, so the report always shows
which pages it read, and every criterion has to say where it came from.
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from rag.step4_retrieve import retrieve_chunks
from rag.step5_prompt import format_context
from rag.step6_generate import CHAT_MODEL, FALLBACK_CHAT_MODEL, MAX_TOKENS, TEMPERATURE

# Models to try in order for the report. A full eligibility report is a long
# structured generation, so this is the call most likely to hit the free tier's
# rate limit. The middle model is a different family, which helps when Groq is
# rate-limiting one of them.
REPORT_MODELS = [CHAT_MODEL, "qwen/qwen3.8-27b", FALLBACK_CHAT_MODEL]


class ReportUnavailable(RuntimeError):
    """Every model refused or failed. The app shows this rather than a stack trace."""


load_dotenv(override=True)

COMPANIES_FILE = Path(__file__).resolve().parent / "data" / "companies.json"

# Eligibility is never stated in one place. It is spread over the notice, the
# instructions to bidders and the document checklist, so we retrieve with
# several targeted queries and pool what comes back - the same approach that
# fixed the tender cards.
ELIGIBILITY_QUERIES = [
    "who is eligible to bid, eligibility criteria for the bidder",
    "required registration, PAN, GST, TIN, income tax returns",
    "experience required, previous contracts, work orders",
    "OEM authorisation letter, manufacturer or authorised dealer",
    "registered office or branch office location required",
    "documents to be submitted along with the bid, checklist",
    "earnest money deposit amount and exemption",
    "grounds on which a bid will be rejected or declared non-responsive",
]


class Criterion(BaseModel):
    """One requirement from the tender, checked against the company."""

    requirement: str = Field(description="The requirement, in under 12 words")
    verdict: str = Field(
        description="Exactly one of: Met, Not met, Unclear. "
        "Use Unclear when the extracts or the company profile do not say."
    )
    reason: str = Field(description="One sentence saying why, quoting the tender where possible")
    tender_page: str = Field(description="The page of the tender this requirement came from, or 'Not stated'")


class EligibilityReport(BaseModel):
    """The whole verdict for one company against one tender."""

    overall: str = Field(
        description="Exactly one of: Likely eligible, Likely not eligible, Needs clarification"
    )
    summary: str = Field(description="Two or three sentences a bid manager could act on")
    criteria: list[Criterion] = Field(description="One entry per requirement found in the extracts")
    missing_documents: list[str] = Field(
        description="Documents the tender asks for that the company profile does not mention"
    )


PROMPT = """You are helping a company decide whether it can bid for a government tender.

Below are extracts from the tender document, followed by the company's profile.

Rules:
- Judge ONLY against the requirements that appear in the tender extracts.
  Do not invent requirements that are not written there.
- If the extracts do not state a requirement clearly, or the company profile
  does not say whether it is met, mark that criterion Unclear rather than guessing.
- Quote the tender wording in the reason where you can.
- Be strict. A missing authorisation letter or a missing registration is "Not met",
  not "Unclear".

Tender extracts:
{context}

Company profile:
{company}

Assess whether this company can bid."""


def load_companies():
    return json.loads(COMPANIES_FILE.read_text())


def company_by_id(company_id):
    for company in load_companies():
        if company["id"] == company_id:
            return company
    return None


def format_company(company):
    """The company profile as plain lines, so it reads well inside the prompt."""
    lines = [f"Company name: {company['name']}"]
    lines += [f"{key}: {value}" for key, value in company["profile"].items()]
    return "\n".join(lines)


def eligibility_chunks(tender):
    """Pool the chunks retrieved by every eligibility query, each one once."""
    chunks, seen = [], set()
    for query in ELIGIBILITY_QUERIES:
        for chunk in retrieve_chunks(query, k=4, tender=tender):
            key = (chunk.metadata["page"], chunk.page_content[:60])
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)
    return chunks


def check(company, tender):
    """Return an EligibilityReport plus the chunks it was based on."""
    chunks = eligibility_chunks(tender)

    prompt = PROMPT.format(
        context=format_context(chunks), company=format_company(company)
    )

    # Same fallback idea as step 6, applied here too: try each model in turn and
    # use the first that answers.
    problems = []
    for model in REPORT_MODELS:
        try:
            llm = init_chat_model(
                model,
                model_provider="groq",
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS * 3,
            )
            report = llm.with_structured_output(EligibilityReport).invoke(prompt)
            if model != REPORT_MODELS[0]:
                print(f"  (answered by the fallback model {model})")
            return report, chunks
        except Exception as error:
            problems.append(f"{model}: {type(error).__name__}")
            print(f"  {model} failed ({type(error).__name__})")

    raise ReportUnavailable(
        "No model could produce the report. This is usually the Groq free tier "
        "rate-limiting after several requests in a row - wait a minute and try "
        "again. Tried: " + "; ".join(problems)
    )


if __name__ == "__main__":
    import sys

    company_id = sys.argv[1] if len(sys.argv) > 1 else "veloce"
    tender = sys.argv[2] if len(sys.argv) > 2 else "niti_sharp_tender"

    company = company_by_id(company_id)
    print(f"{company['name']}  vs  {tender}\n")

    report, chunks = check(company, tender)
    print(f"OVERALL: {report.overall}")
    print(f"{report.summary}\n")
    for item in report.criteria:
        print(f"  [{item.verdict:9}] {item.requirement}  (page {item.tender_page})")
        print(f"              {item.reason}")
    if report.missing_documents:
        print("\nMissing documents:")
        for doc in report.missing_documents:
            print(f"  - {doc}")
    print(f"\nRead {len(chunks)} chunks from pages "
          f"{sorted({c.metadata['page'] for c in chunks})}")
