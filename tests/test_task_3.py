from fastapi.testclient import TestClient

from src.task_3 import citations
from src.task_3 import app as app_module

client = TestClient(app_module.app)

def test_extract_citations():
    answer = "Employees receive leave. [Source: leave_policy.txt]"

    result = citations.extract_citations(answer)

    assert result == ["leave_policy.txt"]

def test_validate_citations():
    answer = "Employees receive leave. [Source: leave_policy.txt]"
    context = "[Source: leave_policy.txt]\nLeave policy details."

    assert citations.validate_citations(answer, context) is True

def test_invalid_citation():
    answer = "Employees receive leave. [Source: unknown.txt]"
    context = "[Source: leave_policy.txt]\nLeave policy details."

    assert citations.validate_citations(answer, context) is False

def test_is_refusal():
    assert citations.is_refusal(
        "I don't have information on that."
    ) is True

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_ask_invalid_api_key(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "APP_API_KEY",
        "correct-key"
    )

    response = client.post(
        "/ask",
        json={"question": "How many casual leave days?"},
        headers={"x-api-key": "wrong-key"}
    )

    assert response.status_code == 401

def test_ask_success(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "APP_API_KEY",
        "correct-key"
    )

    monkeypatch.setattr(
        app_module,
        "answer_with_citations",
        lambda question: {
            "answer": "12 days",
            "citations": ["leave_policy.txt"],
            "valid": True,
            "sources": ["leave_policy.txt"]
        }
    )

    response = client.post(
        "/ask",
        json={"question": "How many casual leave days?"},
        headers={"x-api-key": "correct-key"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "12 days"
    assert response.json()["citation_valid"] is True

def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
        headers={"x-api-key": "correct-key"}
    )

    assert response.status_code == 422