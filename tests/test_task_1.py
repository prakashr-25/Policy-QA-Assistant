from src.task_1 import pipeline

def test_fit_context():
    chunks = [
        "Employees receive 12 casual leave days.",
        "Working hours are 9 AM to 6 PM."
    ]

    metadatas = [
        {"source": "leave_policy.txt"},
        {"source": "attendance_policy.txt"}
    ]

    context = pipeline.fit_context(
        chunks,
        metadatas,
        budget=2000
    )

    assert "leave_policy.txt" in context
    assert "attendance_policy.txt" in context

def test_fit_context_respects_budget():
    chunks = ["This is a test document."]
    metadatas = [{"source": "test.txt"}]

    context = pipeline.fit_context(
        chunks,
        metadatas,
        budget=1
    )

    assert context == ""

def test_answer_returns_generated_answer(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda context, question: "12 days"
    )

    result = pipeline.answer("How many casual leave days?")

    assert result == "12 days"

def test_answer_returns_refusal_for_irrelevant_results(monkeypatch):
    class FakeCollection:
        def query(self, **kwargs):
            return {
                "documents": [["irrelevant content"]],
                "metadatas": [[{"source": "other.txt"}]],
                "distances": [[2.0]]
            }

    monkeypatch.setattr(pipeline, "collection", FakeCollection())

    result = pipeline.answer("What is the stock price?")

    assert result == "I don't have information on that."