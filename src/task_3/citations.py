from src.indexing import collection, embedding_model
from src.generation import generate_answer, extract_citations, validate_citations

DISTANCE_THRESHOLD = 1.2

def answer_with_citations(question, k=4):
    question_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not chunks or min(distances) > DISTANCE_THRESHOLD:
        return {
            "answer": "I don't have information on that.",
            "citations": [],
            "valid": False,
            "sources": []
        }

    context_parts = []

    for chunk, metadata in zip(chunks, metadatas):
        source = metadata.get("source", "Unknown")

        context_parts.append(
            f"[Source: {source}]\n{chunk}"
        )

    context = "\n\n".join(context_parts)

    generated_answer = generate_answer(context, question)

    citations = extract_citations(generated_answer)

    valid = validate_citations(
        generated_answer,
        context
    )

    sources = list(dict.fromkeys(
        metadata.get("source", "Unknown")
        for metadata in metadatas
    ))

    return {
        "answer": generated_answer,
        "citations": citations,
        "valid": valid,
        "sources": sources
    }

def is_refusal(answer):
    answer = answer.lower().replace("’", "'")

    refusal_phrases = [
        "i don't have information on that",
        "documents don't contain any information",
        "documents do not contain any information",
        "information is not available in the context",
        "not provided in the context"
    ]

    return any(
        phrase in answer for phrase in refusal_phrases
    )

def compare_prompts():
    from src.generation import generate_answer_version

    questions = [
        "How many casual leaves are allowed per year?",
        "How many annual leave days do employees receive?",
        "What are the standard working hours?",
        "How many days can employees work from home?",
        "When are salaries paid?",
        "How many public holidays do employees receive?",
        "What employee benefits are provided?",
        "Who won the football World Cup?",
        "What is the company's stock market price?",
        "Who is the CEO's spouse?"
    ]

    unanswerable_questions = {
        "Who won the football World Cup?",
        "What is the company's stock market price?",
        "Who is the CEO's spouse?"
    }

    versions = [
        "v1_basic",
        "v2_context_only",
        "v3_citation"
    ]

    hallucination_counts = {
        version: 0 for version in versions
    }

    for question in questions:
        question_embedding = embedding_model.encode(
            [question]
        ).tolist()

        results = collection.query(
            query_embeddings=question_embedding,
            n_results=4,
            include=["documents", "metadatas"]
        )

        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        context_parts = []

        for chunk, metadata in zip(chunks, metadatas):
            source = metadata.get("source", "Unknown")
            context_parts.append(
                f"[Source: {source}]\n{chunk}"
            )

        context = "\n\n".join(context_parts)

        print("\n" + "=" * 60)
        print("QUESTION:", question)

        for version in versions:
            answer = generate_answer_version(
                context,
                question,
                version
            )

            print(f"\n{version}:")
            print(answer)

            if question in unanswerable_questions:
                if not is_refusal(answer):
                    hallucination_counts[version] += 1

    print("\n" + "=" * 60)
    print("HALLUCINATION COMPARISON")

    for version, count in hallucination_counts.items():
        print(f"{version}: {count} / 3")

if __name__ == "__main__":
    question = "How many annual leave days do employees receive?"

    result = answer_with_citations(question)

    print("Question:", question)
    print("Answer:", result["answer"])
    print("Citations:", result["citations"])
    print("Citation valid:", result["valid"])
    print("Sources:", result["sources"])

    compare_prompts()