import json
import re
import time
from pathlib import Path
from typing import Optional

import ollama
import fasttext
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ---------- CONFIGURATION ----------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bilingual_vectors"              # your actual collection
PARENTS_FILE = "parent_pages/all_parents.json"
EMBED_MODEL = "bge-m3"                            # embedding model (Ollama)
LLM_MODEL = "qwen3:4b"                         # tiny local model for answers
FASTTEXT_MODEL_PATH = "lang_detect_model/lid.176.bin"

# Tenglish keywords (same as in your retrieve.py)
TELUGU_LATIN_WORDS = {
    "ante", "enti", "ela", "unnav", "bagunnara", "emi", "enduku",
    "ekkada", "chesav", "vellava", "naaku", "neeku", "telugu",
    "cheppu", "ardham", "kadu", "avunu", "ledu", "vasthundi",
}
# -----------------------------------

# Load fastText model once
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

def load_parents(parents_file: str) -> dict:
    with open(parents_file, "r", encoding="utf-8") as f:
        parents_list = json.load(f)
    parents_dict = {}
    for p in parents_list:
        doc_id = p["metadata"]["doc_id"]
        page = p["metadata"]["page"]
        first_page = int(str(page).split("-")[0])
        parents_dict[f"{doc_id}::page{first_page}"] = p["page_content"]
    return parents_dict

def detect_query_language(query: str) -> str:
    if re.search(r'[\u0C00-\u0C7F]', query):
        return 'telugu'
    tokens = set(re.findall(r'\b\w+\b', query.lower()))
    if tokens & TELUGU_LATIN_WORDS:
        return 'telugu'
    label, _ = ft_model.predict(query.lower().strip(), k=1)
    lang = label[0].replace('__label__', '')
    return 'telugu' if lang == 'te' else 'english'

def basic_rag(query: str):
    start_time = time.perf_counter()

    # 1. Language detection
    lang = detect_query_language(query)
    print(f"🌐 Detected language: {lang}")

    # 2. Embed the query
    try:
        resp = ollama.embed(model=EMBED_MODEL, input=[query])
        query_vec = resp["embeddings"][0]
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return

    # 3. Search Qdrant (with language filter)
    client = QdrantClient(url=QDRANT_URL)
    lang_filter = Filter(
        must=[FieldCondition(key="language", match=MatchValue(value=lang))]
    )
    try:
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            query_filter=lang_filter,
            limit=5,
            with_payload=True,
        )
    except Exception as e:
        print(f"❌ Qdrant search failed: {e}")
        return

    if not hits.points:
        print("⚠️  No documents found for this query.")
        return

    # 4. Gather parent pages
    parents_dict = load_parents(PARENTS_FILE)
    parent_keys = set()
    for point in hits.points:
        doc_id = point.payload.get("doc_id")
        page = point.payload.get("page")
        if doc_id is not None and page is not None:
            parent_keys.add(f"{doc_id}::page{page}")

    # Build context string from parent pages (deduplicated)
    seen = set()
    context_parts = []
    for key in parent_keys:
        text = parents_dict.get(key)
        if text and text not in seen:
            seen.add(text)
            context_parts.append(text)

    if not context_parts:
        # Fallback to child chunk texts
        context_parts = [p.payload.get("text", "") for p in hits.points[:3]]
    context = "\n\n===== PAGE BREAK =====\n\n".join(context_parts)

    # 5. Generate answer with local LLM
    system_prompt = (
        f"You are a legal assistant for Indian police. Answer the user's question "
        f"ONLY using the provided document context below. Be precise and factual. "
        f"Respond entirely in {'English' if lang == 'english' else 'Telugu'}."
    )
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"},
            ],
            options={"temperature": 0.1},
        )
        answer = response["message"]["content"]
    except Exception as e:
        answer = f"Generation error: {e}"

    elapsed = time.perf_counter() - start_time
    print(f"\n✅ Answer generated in {elapsed:.2f}s")
    print(f"--- Retrieved parent page(s): {len(context_parts)} ---")
    print(answer)


if __name__ == "__main__":
    # Example query – change to your test case
    test_query = "నేర శాస్త్రం అంటే ఏమిటి?"
    print(f"🔍 Testing Basic RAG with: \"{test_query}\"")
    basic_rag(test_query)