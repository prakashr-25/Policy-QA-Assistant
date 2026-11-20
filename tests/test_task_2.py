from src.task_2 import advanced_retrieval as retrieval

def test_tokenize():
    result = retrieval.tokenize("Company Policy 2025")

    assert result == ["company", "policy", "2025"]

def test_reciprocal_rank_fusion():
    rankings = [
        ["id1", "id2"],
        ["id2", "id3"]
    ]

    result = retrieval.reciprocal_rank_fusion(rankings)

    assert result[0] == "id2"
    assert set(result) == {"id1", "id2", "id3"}

def test_bm25_search():
    result = retrieval.bm25_search("casual leave", k=2)

    assert len(result) == 2
    assert all(item in retrieval.ids for item in result)

def test_vector_search(monkeypatch):
    result = retrieval.vector_search("leave policy", k=2)

    assert len(result) == 2

def test_filtered_vector_search():
    result = retrieval.filtered_vector_search(
        "leave policy",
        department="HR",
        year=2025,
        k=2
    )

    assert isinstance(result, list)

def test_rewrite_queries():
    result = retrieval.rewrite_queries("casual leave")

    assert len(result) == 4
    assert result[0] == "casual leave"

def test_hybrid_search():
    result = retrieval.hybrid_search("casual leave", k=2)

    assert len(result) <= 2

def test_query_rewritten_search():
    result = retrieval.query_rewritten_search(
        "casual leave",
        k=2
    )

    assert len(result) <= 2

def test_rerank_search():
    candidates, top_5, reranked = retrieval.rerank_search(
        "casual leave",
        candidate_k=3,
        final_k=2
    )

    assert len(candidates) <= 3
    assert len(top_5) <= 2
    assert set(reranked) == set(candidates)