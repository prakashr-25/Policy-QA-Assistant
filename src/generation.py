import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL")
)

MODEL_NAME = os.getenv("MODEL_NAME")

PROMPT = """Answer ONLY using the context provided below.
Do not use outside knowledge.

Every factual claim must include a citation
using this exact format:
[Source: filename]

Use only filenames present in the provided context.

If the answer is not available in the context, say:
I don't have information on that.

Context:
{context}

Question: {question}

Answer:"""

def generate_answer(context, question):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPT.format(
                    context=context,
                    question=question
                )
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def extract_citations(answer):
    return re.findall(
        r"\[Source:\s*([^\]]+)\]",
        answer
    )

def validate_citations(answer, context):
    citations = extract_citations(answer)

    available_sources = re.findall(
        r"\[Source:\s*([^\]]+)\]",
        context
    )

    if not citations:
        return False

    return all(
        citation in available_sources
        for citation in citations
    )

PROMPTS = {
    "v1_basic": """Answer the question using the context.

Context:
{context}

Question: {question}

Answer:""",

    "v2_context_only": """Answer ONLY using the context provided.
Do not use outside knowledge.
If the answer is unavailable, say:
I don't have information on that.

Context:
{context}

Question: {question}

Answer:""",

    "v3_citation": """Answer ONLY using the context provided.
Do not use outside knowledge.

Every factual claim must include a citation:
[Source: filename]

Use only filenames present in the context.
If the answer is unavailable, say:
I don't have information on that.

Context:
{context}

Question: {question}

Answer:"""
}

def generate_answer_version(context, question, version):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPTS[version].format(
                    context=context,
                    question=question
                )
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()