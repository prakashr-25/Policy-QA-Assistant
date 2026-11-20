import tiktoken

from src.indexing import collection, embedding_model
from src.generation import generate_answer

DISTANCE_THRESHOLD = 1.2

def fit_context(chunks, metadatas, budget):
    encoding = tiktoken.get_encoding("cl100k_base")

    selected = []
    total_tokens = 0

    for chunk, metadata in zip(chunks, metadatas):
        source = metadata.get("source", "Unknown")

        formatted_chunk = (
            f"[Source: {source}]\n"
            f"{chunk}"
        )

        tokens = len(encoding.encode(formatted_chunk))

        if total_tokens + tokens > budget:
            break

        selected.append(formatted_chunk)
        total_tokens += tokens

    return "\n\n".join(selected)

def answer(question, k=4):
    question_embedding = embedding_model.encode([question])

    if hasattr(question_embedding, "tolist"):
        question_embedding = question_embedding.tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not chunks:
        return "I don't have information on that."

    retrieved = [
        (chunk, metadata, distance)
        for chunk, metadata, distance
        in zip(chunks, metadatas, distances)
        if chunk
        and chunk.strip()
        and distance <= DISTANCE_THRESHOLD
    ]

    if not retrieved:
        return "I don't have information on that."

    years = [
        metadata.get("year")
        for _, metadata, _ in retrieved
        if metadata.get("year") is not None
    ]

    if years:
        latest_year = max(years)

        retrieved = [
            item for item in retrieved
            if item[1].get("year") == latest_year
        ]

    filtered_chunks = [item[0] for item in retrieved]
    filtered_metadatas = [item[1] for item in retrieved]

    context = fit_context(
        filtered_chunks,
        filtered_metadatas,
        budget=2000
    )

    if not context:
        return "I don't have information on that."

    return generate_answer(context, question)

if __name__ == "__main__":
    questions = [
        "How many casual leaves per year?",
        "How many annual leave days do employees receive?",
        "What are the standard working hours?",
        "How many days can employees work from home?",
        "When are salaries paid?",
        "How many public holidays do employees receive?",
        "Who won the football world cup?",
        "What is the company's stock market price?",
        "Who is the CEO's spouse?",
        "What is the company's office electricity bill?"
    ]

    for question in questions:
        print("\nQ:", question)
        print("A:", answer(question))