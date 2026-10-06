"""Step 6: send the prompt to the LLM and get the answer.

In  : a question
Out : the answer, plus the chunks it was based on

There are two ways to call the model in this file, on purpose.

1. generate_answer() is the class way: LangChain's init_chat_model and an
   LCEL chain, prompt | llm | StrOutputParser(). This is the main path.

2. answer_with_fallback() is the resilience way, using LiteLLM. It tries a
   primary model, and if that call raises, it sends the same prompt to a
   second model. We added this because our teacher liked the fallback idea
   in our previous project, and because a free API tier can rate-limit in
   the middle of a demo.

Both send the exact same prompt built in step 5.
"""

import os

import litellm
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from rag.step4_retrieve import TOP_K, retrieve_chunks
from rag.step5_prompt import RAG_PROMPT, format_context

load_dotenv(override=True)

# Reading a tender is a factual task, not a creative one. We want the same
# question to give the same answer, so the temperature is 0. (The class
# notebook on generation parameters puts deterministic, extraction-style
# tasks at the low end of the range.)
TEMPERATURE = 0.0

# gpt-oss models think before they answer, and that thinking is counted in
# the token budget. If max_tokens is too small the whole budget is used up
# thinking and the reply comes back empty, so we leave plenty of room.
MAX_TOKENS = 1200

CHAT_MODEL = "openai/gpt-oss-120b"

# For LiteLLM the provider is part of the model name.
PRIMARY_MODEL = "groq/openai/gpt-oss-120b"
FALLBACK_MODEL = "groq/openai/gpt-oss-20b"


def generate_answer(question, k=TOP_K, tender=None):
    """The main path: retrieve, build the prompt, call the LLM with LCEL."""
    chunks = retrieve_chunks(question, k=k, tender=tender)

    llm = init_chat_model(
        CHAT_MODEL,
        model_provider="groq",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    # The LCEL chain from the class notebook: the prompt is filled in, passed
    # to the model, and the model's reply is turned into a plain string.
    chain = RAG_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "context": format_context(chunks),
        "question": question,
    })
    return answer, chunks


def ask_llm_with_fallback(prompt_text):
    """Try the primary model. If the call fails, use the fallback model.

    Returns the answer and the name of the model that actually produced it,
    so the app can show which one answered.
    """
    messages = [{"role": "user", "content": prompt_text}]
    try:
        response = litellm.completion(
            model=PRIMARY_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content, PRIMARY_MODEL
    except Exception as error:
        print(f"  primary model failed ({type(error).__name__}), "
              f"switching to {FALLBACK_MODEL}")
        response = litellm.completion(
            model=FALLBACK_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content, FALLBACK_MODEL


def answer_with_fallback(question, k=TOP_K, tender=None):
    """Same pipeline as generate_answer, but the LLM call goes through LiteLLM."""
    chunks = retrieve_chunks(question, k=k, tender=tender)
    prompt_text = RAG_PROMPT.format(
        context=format_context(chunks),
        question=question,
    )
    answer, model_used = ask_llm_with_fallback(prompt_text)
    return answer, chunks, model_used


def show_sources(chunks):
    """A short list of where the answer came from."""
    return "\n".join(
        f"  - {c.metadata['tender']}, page {c.metadata['page']}" for c in chunks
    )


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env.")

    question = "What is the EMD amount for the home lift tender?"
    print(f"Question: {question}\n")

    answer, chunks = generate_answer(question)
    print("Answer:", answer)
    print("\nSources:")
    print(show_sources(chunks))
