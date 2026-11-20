import os
import sys
import re
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

GOLDEN_FILE = ROOT / "data" / "golden.jsonl"
RESULT_FILE = ROOT / "data" / "ragas_scores.jsonl"

from src.task_1.pipeline import (
    answer,
    collection,
    embedding_model,
    DISTANCE_THRESHOLD,
)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

def run_pipeline(question, reference, category):
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    retrieved = collection.query(
        query_embeddings=question_embedding,
        n_results=4,
        include=["documents", "metadatas", "distances"],
    )

    documents = retrieved["documents"][0]
    distances = retrieved["distances"][0]
    metadatas = retrieved["metadatas"][0]

    contexts = [
        document
        for document, distance in zip(documents, distances)
        if document
        and document.strip()
        and distance <= DISTANCE_THRESHOLD
    ]

    response = answer(question)

    print(
        f"Retrieved: {len(documents)}, "
        f"Contexts supplied to RAGAS: {len(contexts)}"
    )

    return {
        "user_input": question,
        "response": response,
        "retrieved_contexts": contexts,
        "reference": reference,
        "category": category,
        "retrieved_sources": metadatas,
        "distances": distances,
    }

async def score_with_retry(metric, arguments, metric_name):
    for attempt in range(1, 4):
        try:
            result = await metric.ascore(**arguments)
            return float(result.value)

        except Exception as error:
            message = str(error)
            print(
                f"{metric_name} attempt {attempt}/3 failed: "
                f"{message}"
            )

            if attempt == 3:
                return None

            if "429" in message:
                match = re.search(
                    r"try again in\s+(?:(\d+)m)?([\d.]+)s",
                    message,
                    re.IGNORECASE,
                )

                if match:
                    minutes = int(match.group(1) or 0)
                    seconds = float(match.group(2))
                    wait_time = int(minutes * 60 + seconds) + 5
                else:
                    wait_time = 300

            elif "503" in message:
                wait_time = 30

            elif "json_validate_failed" in message:
                wait_time = 10

            else:
                return None

            print(f"Retrying {metric_name} in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    return None

async def main():
    print("\nLoading golden dataset...")

    golden = load_jsonl(GOLDEN_FILE)

    if len(golden) != 15:
        raise ValueError(
            f"Expected 15 golden questions, found {len(golden)}"
        )

    print("Golden questions:", len(golden))

    dataset = []

    for index, item in enumerate(golden, 1):
        question = item["question"]
        reference = item["answer"]
        category = item.get("category", "answerable")

        print(f"\nProcessing {index}/15: {question}")

        result = run_pipeline(
            question,
            reference,
            category,
        )
        dataset.append(result)

    client = AsyncOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    llm = llm_factory(
        os.getenv("MODEL_NAME"),
        client=client,
        temperature=0,
        max_tokens=2048,
    )

    embeddings = embedding_factory(
        "huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

    metrics = {
        "faithfulness": Faithfulness(llm=llm),

        "answer_relevancy": AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=1,
        ),

        "context_precision": ContextPrecision(llm=llm),

        "context_recall": ContextRecall(llm=llm),
    }

    results = []

    print("\nStarting RAGAS evaluation...")

    for index, row in enumerate(dataset, 1):
        print(f"\nEvaluating {index}/15")

        scores = {
            "question": row["user_input"],
            "category": row["category"],
        }

        for name, metric in metrics.items():

            if (
                not row["retrieved_contexts"]
                and name in [
                    "faithfulness",
                    "context_precision",
                    "context_recall",
                ]
            ):
                scores[name] = None
                print(f"{name}: Skipped (no relevant contexts)")
                continue

            if name == "faithfulness":
                arguments = {
                    "user_input": row["user_input"],
                    "response": row["response"],
                    "retrieved_contexts": row["retrieved_contexts"],
                }

            elif name == "answer_relevancy":
                arguments = {
                    "user_input": row["user_input"],
                    "response": row["response"],
                }

            else:
                arguments = {
                    "user_input": row["user_input"],
                    "reference": row["reference"],
                    "retrieved_contexts": row["retrieved_contexts"],
                }

            value = await score_with_retry(
                metric,
                arguments,
                name,
            )

            scores[name] = value

            if value is not None:
                print(f"{name}: {value:.4f}")
            else:
                print(f"{name}: Evaluation failed after retries")

            await asyncio.sleep(13)

        results.append(scores)
        save_jsonl(RESULT_FILE, results)

    print("\n" + "=" * 65)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 65)

    for name in metrics:
        values = [
            row[name]
            for row in results
            if row.get(name) is not None
        ]

        if values:
            average = sum(values) / len(values)
            print(
                f"{name:<25} {average:.4f} "
                f"({len(values)}/15 valid)"
            )
        else:
            print(f"{name:<25} No valid scores")

    unanswerable = [
        row for row in dataset
        if row["category"] == "unanswerable"
    ]

    correct_refusals = sum(
        row["response"].strip().lower()
        == row["reference"].strip().lower()
        for row in unanswerable
    )

    print(
        f"{'Refusal correctness':<25} "
        f"{correct_refusals}/{len(unanswerable)}"
    )

    print("=" * 65)

    save_jsonl(RESULT_FILE, results)

if __name__ == "__main__":
    asyncio.run(main())