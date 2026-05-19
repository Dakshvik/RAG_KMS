import json
import uuid
from pathlib import Path
from collections import defaultdict

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ---------- CONFIGURATION ----------
CHUNKS_DIR = "extracted_chunks_1"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bilingual_vectors"      # single collection for both languages
EMBED_MODEL = "bge-m3"
VECTOR_SIZE = 1024
BATCH_EMBED_SIZE = 10                    # number of texts to send to Ollama at once
# -----------------------------------

def load_all_chunks(chunks_dir: str):
    """Yield all valid, non‑empty chunk dicts from JSON files."""
    json_files = sorted(Path(chunks_dir).glob("*.json"))
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                content = item.get("page_content", "").strip()
                if not content:
                    continue
                yield item

def get_existing_point_ids(client: QdrantClient, collection_name: str) -> set:
    """Retrieve all point IDs already stored in the collection."""
    existing = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in points:
            existing.add(point.id)
        if next_offset is None:
            break
        offset = next_offset
    return existing

def embed_chunks(chunks, embed_model, existing_ids: set):
    """
    Embed new chunks in batches, skipping any whose point ID already exists.
    Returns list of PointStruct ready to upsert.
    """
    # Group all chunks by page for potential fallback
    page_chunks = defaultdict(list)
    for c in chunks:
        meta = c["metadata"]
        page_chunks[(meta["doc_id"], meta["page"])].append(c)

    new_chunks = []            # chunks that need embedding
    for c in chunks:
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, c["metadata"]["chunk_id"]))
        if pid not in existing_ids:
            new_chunks.append(c)

    if not new_chunks:
        print("No new chunks to embed.")
        return []

    print(f"Embedding {len(new_chunks)} new chunks (out of {len(chunks)} total).")

    points = []
    # Track which pages have at least one successful embed
    pages_with_success = set()
    pages_with_failure = set()

    # Process in batches
    for i in range(0, len(new_chunks), BATCH_EMBED_SIZE):
        batch = new_chunks[i:i + BATCH_EMBED_SIZE]
        texts = [c["page_content"] for c in batch]

        try:
            # Batch embed via Ollama
            resp = ollama.embed(model=embed_model, input=texts)
            vectors = resp["embeddings"]
        except Exception as e:
            print(f"  ⚠️ Batch embedding failed: {e}")
            # Fall back to one‑by‑one for this batch
            vectors = []
            for chunk, text in zip(batch, texts):
                try:
                    res = ollama.embed(model=embed_model, input=[text])
                    vectors.append(res["embeddings"][0])
                except Exception as inner_e:
                    print(f"    🚩 Skipping chunk {chunk['metadata']['chunk_id']}: {inner_e}")
                    vectors.append(None)
            # Filter out failed ones
            new_batch = [c for c, v in zip(batch, vectors) if v is not None]
            vectors = [v for v in vectors if v is not None]
            batch = new_batch

        for chunk, vec in zip(batch, vectors):
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["metadata"]["chunk_id"]))
            points.append(PointStruct(
                id=pid,
                vector=vec,
                payload={"text": chunk["page_content"], **chunk["metadata"]}
            ))
            doc_id = chunk["metadata"]["doc_id"]
            page = chunk["metadata"]["page"]
            pages_with_success.add((doc_id, page))

        # Mark any chunk that failed in the one‑by‑one fallback as a page failure
        for chunk in batch:
            if chunk not in zip(batch, vectors):  # crude; better to track failures explicitly
                pass
        # We'll recompute failures after loop using success set
        print(f"  ... processed {min(i + BATCH_EMBED_SIZE, len(new_chunks))}/{len(new_chunks)}")

    # Determine pages that had only failures (no successful child embed)
    for c in new_chunks:
        doc_id = c["metadata"]["doc_id"]
        page = c["metadata"]["page"]
        if (doc_id, page) not in pages_with_success:
            pages_with_failure.add((doc_id, page))

    # Create fallback points for pages with zero successful child embeddings
    if pages_with_failure:
        print(f"\n🔧 Creating fallback vectors for {len(pages_with_failure)} page(s) with no successful child embeds.")
        for (doc_id, page) in pages_with_failure:
            chunks_of_page = page_chunks[(doc_id, page)]
            full_text = " ".join([c["page_content"] for c in chunks_of_page])
            if not full_text.strip():
                continue
            try:
                res = ollama.embed(model=embed_model, input=[full_text])
                vec = res["embeddings"][0]
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}::page{page}::fallback"))
                points.append(PointStruct(
                    id=pid,
                    vector=vec,
                    payload={
                        "text": full_text,
                        "doc_id": doc_id,
                        "page": page,
                        "language": chunks_of_page[0]["metadata"]["language"],
                        "is_fallback": True,
                    }
                ))
                print(f"  ✅ Fallback created for doc_id={doc_id}, page={page}")
            except Exception as e:
                print(f"  ❌ Fallback failed for doc_id={doc_id}, page={page}: {e}")

    return points

def main():
    client = QdrantClient(url=QDRANT_URL)

    # Create collection if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    # Load all chunks from JSON files
    all_chunks = list(load_all_chunks(CHUNKS_DIR))
    print(f"Found {len(all_chunks)} valid child chunks on disk.")

    if not all_chunks:
        print("No chunks to process. Exiting.")
        return

    # Get existing point IDs to skip already embedded chunks
    print("Checking existing embeddings...")
    existing_ids = get_existing_point_ids(client, COLLECTION_NAME)
    print(f"  {len(existing_ids)} chunks already indexed.")

    # Embed only new chunks
    new_points = embed_chunks(all_chunks, EMBED_MODEL, existing_ids)

    if new_points:
        # Upsert in batches
        BATCH_UPLOAD = 200
        for i in range(0, len(new_points), BATCH_UPLOAD):
            batch = new_points[i:i + BATCH_UPLOAD]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            print(f"  Upserted batch {i//BATCH_UPLOAD + 1}/{(len(new_points)-1)//BATCH_UPLOAD + 1}")
        print(f"✅ Upserted {len(new_points)} new points.")
    else:
        print("No new points to upsert.")

    print("Done.")

if __name__ == "__main__":
    main()