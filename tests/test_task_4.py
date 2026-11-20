import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_FILE = ROOT / "data" / "golden.jsonl"

def load_golden_dataset():
    with open(GOLDEN_FILE, "r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

def test_golden_dataset_exists():
    assert GOLDEN_FILE.exists()

def test_golden_dataset_has_15_questions():
    data = load_golden_dataset()

    assert len(data) == 15

def test_golden_dataset_required_fields():
    data = load_golden_dataset()

    required_fields = {
        "question",
        "answer",
        "source",
        "page"
    }

    for item in data:
        assert required_fields.issubset(item.keys())

def test_golden_dataset_has_no_empty_questions():
    data = load_golden_dataset()

    for item in data:
        assert item["question"].strip()

def test_golden_dataset_has_no_empty_answers():
    data = load_golden_dataset()

    for item in data:
        assert item["answer"].strip()