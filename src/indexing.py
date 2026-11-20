import re
from pathlib import Path
from hashlib import sha256

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "documents"

client = chromadb.PersistentClient(path=str(ROOT / "index"))
collection = client.get_or_create_collection("company_policies")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents():
    documents = []

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        department = re.search(r"Department:\s*(.+)", text)
        year = re.search(r"Year:\s*(\d{4})", text)

        metadata = {
            "source": file_path.name,
            "department": department.group(1).strip() if department else "Unknown",
            "year": int(year.group(1)) if year else 0
        }

        documents.append((text, metadata))

    return documents

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def index_documents():
    documents = load_documents()

    ids = []
    chunks = []
    metadatas = []

    for text, metadata in documents:
        text_chunks = chunk_text(text)

        for i, chunk in enumerate(text_chunks):
            chunk_id = sha256(
                f"{metadata['source']}:{i}".encode()
            ).hexdigest()

            ids.append(chunk_id)
            chunks.append(chunk)
            metadatas.append({
                **metadata,
                "chunk_index": i
            })

    embeddings = embedding_model.encode(chunks).tolist()

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Total stored chunks: {collection.count()}")


if __name__ == "__main__":
    index_documents()