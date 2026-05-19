import json
import uuid
import re
from pathlib import Path
from collections import defaultdict

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector,
)
from fastembed import SparseTextEmbedding

# ---------- CONFIGURATION ----------
CHUNKS_DIR = "extracted_chunks_1"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bilingual_hybrid"
EMBED_MODEL = "bge-m3"
SPARSE_MODEL = "Qdrant/bm25"
VECTOR_SIZE = 1024
BATCH_SIZE = 10
# -----------------------------------

sparse_model = SparseTextEmbedding(SPARSE_MODEL)

def sanitize_text(text: str) -> str:
    """
    Keep ONLY:
    - Printable ASCII (letters, digits, punctuation)
    - Telugu Unicode block (0C00-0C7F)
    - Common whitespace (space, newline, tab)
    Everything else is removed.
    """
    cleaned = []
    for ch in text:
        cp = ord(ch)
        # Telugu block
        if 0x0C00 <= cp <= 0x0C7F:
            cleaned.append(ch)
        # Printable ASCII (space to ~) plus newline, tab, carriage return
        elif 32 <= cp <= 126 or cp in (9, 10, 13):
            cleaned.append(ch)
        # else: discard the character
    result = ''.join(cleaned)
    # Normalize multiple spaces
    result = re.sub(r'\s+', ' ', result)
    return result.strip()

def load_all_chunks(chunks_dir: str):
    json_files = sorted(Path(chunks_dir).glob("*.json"))
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                content = item.get("page_content", "").strip()
                if not content:
                    continue
                # Sanitize here so all downstream processing receives clean text
                item["page_content"] = sanitize_text(content)
                if not item["page_content"]:
                    continue
                yield item

def get_existing_ids(client, collection_name):
    existing = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name, limit=1000, offset=offset,
            with_payload=False, with_vectors=False,
        )
        for p in points:
            existing.add(p.id)
        if next_offset is None:
            break
        offset = next_offset
    return existing

def main():
    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )
        print(f"Created hybrid collection: {COLLECTION_NAME}")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    all_chunks = list(load_all_chunks(CHUNKS_DIR))
    print(f"Total valid chunks (after sanitization): {len(all_chunks)}")

    existing_ids = get_existing_ids(client, COLLECTION_NAME)
    print(f"Already indexed: {len(existing_ids)} points")

    new_chunks = []
    for chunk in all_chunks:
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["metadata"]["chunk_id"]))
        if pid not in existing_ids:
            new_chunks.append(chunk)

    if not new_chunks:
        print("No new chunks to index.")
        return

    print(f"Embedding {len(new_chunks)} new chunks...")
    points = []
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i+BATCH_SIZE]
        texts = [c["page_content"] for c in batch]

        # Dense embeddings
        try:
            dense_resp = ollama.embed(model=EMBED_MODEL, input=texts)
            dense_vecs = dense_resp["embeddings"]
        except Exception as e:
            print(f"Dense batch failed: {e}. Trying one‑by‑one sanitized...")
            # Fallback: embed one‑by‑one (already sanitized, so rarely fails)
            dense_vecs = []
            for text in texts:
                try:
                    res = ollama.embed(model=EMBED_MODEL, input=[text])
                    dense_vecs.append(res["embeddings"][0])
                except Exception as inner_e:
                    print(f"  Skipping chunk: {inner_e}")
                    dense_vecs.append(None)
            # Filter out failed ones
            valid = [(ch, dv) for ch, dv in zip(batch, dense_vecs) if dv is not None]
            batch, dense_vecs = zip(*valid) if valid else ([], [])

        # Sparse embeddings (BM25)
        sparse_embs = list(sparse_model.embed(texts))

        for chunk, dvec, svec in zip(batch, dense_vecs, sparse_embs):
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["metadata"]["chunk_id"]))
            qdrant_svec = SparseVector(
                indices=svec.indices.tolist(),
                values=svec.values.tolist(),
            )
            points.append(PointStruct(
                id=pid,
                vector={
                    "dense": dvec,
                    "sparse": qdrant_svec,
                },
                payload={
                    "text": chunk["page_content"],
                    **chunk["metadata"],
                },
            ))

        if len(points) >= 200:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"  Upserted {len(points)} points")
            points = []

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Upserted final {len(points)} points")

    print("Indexing complete.")

if __name__ == "__main__":
    main()