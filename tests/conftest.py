import sys
import types
import re

class FakeCollection:
    def get(self, include=None):
        return {
            "ids": ["id1", "id2", "id3"],
            "documents": [
                "Employees receive 12 casual leave days.",
                "Working hours are 9 AM to 6 PM.",
                "Employees can work remotely."
            ],
            "metadatas": [
                {"source": "leave_policy.txt", "department": "HR", "year": 2025},
                {"source": "attendance_policy.txt", "department": "HR", "year": 2025},
                {"source": "work_from_home.txt", "department": "IT", "year": 2025}
            ]
        }

    def query(self, **kwargs):
        return {
            "ids": [["id1", "id2"]],
            "documents": [[
                "Employees receive 12 casual leave days.",
                "Working hours are 9 AM to 6 PM."
            ]],
            "metadatas": [[
                {"source": "leave_policy.txt"},
                {"source": "attendance_policy.txt"}
            ]],
            "distances": [[0.3, 0.5]]
        }

class FakeEmbeddingModel:
    def encode(self, texts):
        return [[0.1, 0.2] for _ in texts]

indexing = types.ModuleType("src.indexing")
indexing.collection = FakeCollection()
indexing.embedding_model = FakeEmbeddingModel()

sys.modules["src.indexing"] = indexing

generation = types.ModuleType("src.generation")

def generate_answer(context, question):
    return "Employees receive 12 casual leave days. [Source: leave_policy.txt]"

def extract_citations(answer):
    return re.findall(r"\[Source:\s*([^\]]+)\]", answer)

def validate_citations(answer, context):
    citations = extract_citations(answer)
    available = re.findall(r"\[Source:\s*([^\]]+)\]", context)
    return bool(citations) and all(c in available for c in citations)

def generate_answer_version(context, question, version):
    return "Test answer"

generation.generate_answer = generate_answer
generation.extract_citations = extract_citations
generation.validate_citations = validate_citations
generation.generate_answer_version = generate_answer_version

sys.modules["src.generation"] = generation

sentence_transformers = types.ModuleType("sentence_transformers")

class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.9] * len(pairs)

sentence_transformers.CrossEncoder = FakeCrossEncoder
sys.modules["sentence_transformers"] = sentence_transformers