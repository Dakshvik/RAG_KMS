import json
import uuid
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
COLLECTION_NAME = "bilingual_hybrid"    # new collection for hybrid vectors
EMBED_MODEL = "bge-m3"                 # dense embedding model (Ollama)
SPARSE_MODEL = "Qdrant/bm25"           # sparse keyword model
VECTOR_SIZE = 1024
BATCH_SIZE = 10                        # texts per Ollama call
# -----------------------------------

sparse_model = SparseTextEmbedding(SPARSE_MODEL)

def load_all_chunks(chunks_dir: str):
    json_files = sorted(Path(chunks_dir).glob("*.json"))
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                content = item.get("page_content", "").strip()
                if not content:
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

    # Create collection with both dense and sparse vectors
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
    print(f"Total valid chunks: {len(all_chunks)}")

    existing_ids = get_existing_ids(client, COLLECTION_NAME)
    print(f"Already indexed: {len(existing_ids)} points")

    # Filter only new chunks
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
            print(f"Dense embedding failed for batch {i//BATCH_SIZE}: {e}")
            # Fall back to one-by-one
            dense_vecs = []
            for text in texts:
                try:
                    res = ollama.embed(model=EMBED_MODEL, input=[text])
                    dense_vecs.append(res["embeddings"][0])
                except:
                    dense_vecs.append(None)
            # Filter out failed
            valid_indices = [j for j, v in enumerate(dense_vecs) if v is not None]
            batch = [batch[j] for j in valid_indices]
            texts = [texts[j] for j in valid_indices]
            dense_vecs = [dense_vecs[j] for j in valid_indices]

        # Sparse embeddings (BM25)
        sparse_embs = list(sparse_model.embed(texts))  # generator to list

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

        # Upsert in sub-batches of 200
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