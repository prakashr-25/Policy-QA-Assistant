import re
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.indexing import collection, embedding_model

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def tokenize(text):
    return re.findall(r"\w+", text.lower())

stored = collection.get(include=["documents", "metadatas"])

documents = stored["documents"]
metadatas = stored["metadatas"]
ids = stored["ids"]

id_to_source = {
    doc_id: metadata["source"]
    for doc_id, metadata in zip(ids, metadatas)
}

id_to_document = {
    doc_id: document
    for doc_id, document in zip(ids, documents)
}

tokenized_documents = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

def bm25_search(question, k=3):
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )
    return [ids[i] for i in ranked[:k]]

def vector_search(question, k=3):
    embedding = embedding_model.encode([question])

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=k
    )

    return results["ids"][0]

def filtered_vector_search(question, department, year, k=3):
    embedding = embedding_model.encode([question])
    
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=k,
        where={
            "$and": [
                {"department": {"$eq": department}},
                {"year": {"$eq": year}}
            ]
        }
    )

    return results["ids"][0]

def hit_at_3(search_function, question, expected_source):
    results = search_function(question, k=3)

    return int(any(
        id_to_source[doc_id] == expected_source
        for doc_id in results
    ))

questions = [
    ("casual leave days per year", "leave_policy.txt"),
    ("working hours 9 AM to 6 PM", "attendance_policy.txt"),
    ("work from home 2 days per week", "work_from_home.txt"),
    ("salary last working day", "salary_policy.txt"),
    ("business travel expense claims", "travel_policy.txt"),
    ("multi-factor authentication", "IT_security_policy.txt"),

    ("time off employees can take when sick", "leave_policy.txt"),
    ("procedure for recording employee arrival", "attendance_policy.txt"),
    ("remote working arrangement", "work_from_home.txt"),
    ("how employees receive their monthly pay", "salary_policy.txt"),
    ("medical protection provided by employer", "employee_benefits.txt"),
    ("rules against unfair treatment at work", "code_of_conduct.txt")
]

def reciprocal_rank_fusion(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + (
                1 / (k + rank)
            )

    return sorted(
        scores,
        key=scores.get,
        reverse=True
    )

def hybrid_search(question, k=3):
    bm25_results = bm25_search(question, k=10)
    vector_results = vector_search(question, k=10)

    return reciprocal_rank_fusion(
        [bm25_results, vector_results]
    )[:k]

def rewrite_queries(question):
    return [
        question,
        f"What is the company policy regarding {question}?",
        f"What rules or procedures apply to {question}?",
        f"How does the organization handle {question}?"
    ]

def query_rewritten_search(question, k=3):
    queries = rewrite_queries(question)

    rankings = []

    for rewritten_query in queries:
        rankings.append(bm25_search(rewritten_query, k=10))
        rankings.append(vector_search(rewritten_query, k=10))

    return reciprocal_rank_fusion(rankings)[:k]

def rerank_search(question, candidate_k=50, final_k=5):
    bm25_results = bm25_search(question, k=candidate_k)
    vector_results = vector_search(question, k=candidate_k)

    candidates = reciprocal_rank_fusion(
        [bm25_results, vector_results]
    )[:candidate_k]

    pairs = [
        (question, id_to_document[doc_id])
        for doc_id in candidates
    ]

    scores = cross_encoder.predict(pairs)

    reranked = [
        doc_id
        for doc_id, score in sorted(
            zip(candidates, scores),
            key=lambda item: item[1],
            reverse=True
        )
    ]

    return candidates, reranked[:final_k], reranked

def evaluate_query_rewriting():
    methods = {
        "BM25": bm25_search,
        "Vector": vector_search,
        "Hybrid": hybrid_search,
        "Query Rewritten": query_rewritten_search
    }

    hits = {name: 0 for name in methods}
    total_time = {name: 0.0 for name in methods}

    print("\nQuery Rewriting Evaluation")

    for question, expected_source in questions:

        print(f"\nQuestion: {question}")

        for name, search_function in methods.items():
            start = time.perf_counter()

            results = search_function(question, k=3)

            elapsed = time.perf_counter() - start
            total_time[name] += elapsed

            hit = int(any(
                id_to_source[doc_id] == expected_source
                for doc_id in results
            ))

            hits[name] += hit

            if name == "Query Rewritten":
                print(f"Query Rewritten Hit@3: {hit}")

    print("\nQuery Rewriting Comparison")

    for name in methods:
        print(
            f"{name}: "
            f"Hit@3 = {hits[name]}/12 "
            f"({hits[name] / 12:.2f}) | "
            f"Average latency = "
            f"{total_time[name] / 12 * 1000:.2f} ms"
        )

def evaluate_metadata_filter():
    question = "How many casual leave days per year?"

    print("\nMetadata Filtering Evaluation")

    all_results = vector_search(question, k=10)

    filtered_results = filtered_vector_search(
        question,
        department="HR",
        year=2025,
        k=10
    )

    print("\nBefore metadata filtering:")

    for doc_id in all_results:
        print(id_to_source[doc_id])

    print("\nAfter metadata filtering:")

    for doc_id in filtered_results:
        print(id_to_source[doc_id])

    outdated_present = any(
        id_to_source[doc_id] == "leave_policy_2023.txt"
        for doc_id in filtered_results
    )

    print("\nOutdated document removed:", not outdated_present)

def evaluate_reranker():
    print("\nCross-Encoder Reranking (Top 50 -> Top 5)")

    for question, expected_source in questions[:10]:

        candidates, top_5, reranked = rerank_search(question)

        before_rank = next(
            (
                rank
                for rank, doc_id in enumerate(candidates, start=1)
                if id_to_source[doc_id] == expected_source
            ),
            None
        )

        after_rank = next(
            (
                rank
                for rank, doc_id in enumerate(reranked, start=1)
                if id_to_source[doc_id] == expected_source
            ),
            None
        )

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected_source}")
        print(f"Rank before reranking: {before_rank}")
        print(f"Rank after reranking: {after_rank}")
        print(f"Retrieved in top 5: {expected_source in [id_to_source[doc_id] for doc_id in top_5]}")

def evaluate_retrieval():
    bm25_hits = 0
    vector_hits = 0
    hybrid_hits = 0

    for question, expected_source in questions:
        bm25_hit = hit_at_3(
            bm25_search, question, expected_source
        )
        vector_hit = hit_at_3(
            vector_search, question, expected_source
        )
        hybrid_hit = hit_at_3(
            hybrid_search, question, expected_source
        )

        bm25_hits += bm25_hit
        vector_hits += vector_hit
        hybrid_hits += hybrid_hit

        print(f"\nQuestion: {question}")
        print(f"Expected: {expected_source}")
        print(f"BM25 Hit@3: {bm25_hit}")
        print(f"Vector Hit@3: {vector_hit}")
        print(f"Hybrid Hit@3: {hybrid_hit}")

    print("\nHit@3 Comparison--")
    print(f"BM25: {bm25_hits}/12 = {bm25_hits / 12:.2f}")
    print(f"Vector: {vector_hits}/12 = {vector_hits / 12:.2f}")
    print(f"Hybrid: {hybrid_hits}/12 = {hybrid_hits / 12:.2f}")

if __name__ == "__main__":
    evaluate_retrieval()
    evaluate_reranker()
    evaluate_query_rewriting()
    evaluate_metadata_filter()