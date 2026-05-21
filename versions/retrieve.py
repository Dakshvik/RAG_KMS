import json
import re
from pathlib import Path
from typing import List, Optional, Union

import ollama
import fasttext
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document

# ---------- CONFIGURATION ----------
PARENTS_FILE = "parent_pages/all_parents.json"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bilingual_hybrid_final"          # adjust if needed
EMBED_MODEL = "bge-m3"
FASTTEXT_MODEL_PATH = "lang_detect_model\lid.176.bin"

# Optional: Tenglish keywords (Latin Telugu words)
TELUGU_LATIN_WORDS = {
    "ante", "enti", "ela", "unnav", "bagunnara", "emi", "enduku",
    "ekkada", "chesav", "vellava", "naaku", "neeku", "telugu",
    "cheppu", "ardham", "kadu", "avunu", "ledu", "vasthundi",
}
# -----------------------------------

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
    # Native Telugu script → Telugu
    if re.search(r'[\u0C00-\u0C7F]', query):
        return 'telugu'
    # Check Latin Telugu keywords (Tenglish)
    tokens = set(re.findall(r'\b\w+\b', query.lower()))
    if tokens & TELUGU_LATIN_WORDS:
        return 'telugu'
    # Fallback to fastText
    label, _ = ft_model.predict(query.lower().strip(), k=1)
    lang = label[0].replace('__label__', '')
    return 'telugu' if lang == 'te' else 'english'


def build_retriever():
    client = QdrantClient(url=QDRANT_URL)
    parents_dict = load_parents(PARENTS_FILE)

    def retrieve_full_pages(
        query: str,
        k: int = 4,
        language: Optional[str] = None,
    ) -> List[Document]:
        """
        Returns both:
          - Child chunks that matched the query (with metadata indicating `chunk_type = "child"`)
          - Full parent pages of those matches (with `chunk_type = "parent"`)
        """
        if language is None:
            language = detect_query_language(query)

        # 1. Embed query
        try:
            resp = ollama.embed(model=EMBED_MODEL, input=[query])
            query_vector = resp["embeddings"][0]
        except Exception as e:
            print(f"❌ Failed to embed query: {e}")
            return []

        # 2. Language filter
        lang_filter = Filter(
            must=[FieldCondition(key="language", match=MatchValue(value=language))]
        )

        # 3. Vector search
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=lang_filter,
            limit=k,
            with_payload=True,
        )

        if not response.points:
            return []

        # 4. Collect child documents and unique parent keys
        child_docs = []
        parent_keys = set()
        for point in response.points:
            payload = point.payload
            text = payload.get("text", "")
            doc_id = payload.get("doc_id")
            page = payload.get("page")
            if doc_id is None or page is None:
                continue

            # Child chunk
            child_meta = {
                "doc_id": doc_id,
                "page": page,
                "language": language,
                "chunk_type": "child",
                "score": point.score,
                "source_key": f"{doc_id}::page{page}",
                # preserve original metadata from the payload (optional)
                "chunk_id": payload.get("chunk_id", ""),
            }
            child_docs.append(Document(page_content=text, metadata=child_meta))
            parent_keys.add(f"{doc_id}::page{page}")

        # 5. Build parent documents from the unique pages
        parent_docs = []
        for key in parent_keys:
            page_text = parents_dict.get(key)
            if page_text:
                parent_meta = {
                    "source_key": key,
                    "chunk_type": "parent",
                    "language": language,
                }
                parent_docs.append(Document(page_content=page_text, metadata=parent_meta))

        # 6. Return children first (so LLM sees precise matches), then full pages
        return child_docs + parent_docs

    return retrieve_full_pages


# ---------- example usage ----------
if __name__ == "__main__":
    retrieve = build_retriever()

    print("=== English ===")
    for doc in retrieve("What is BNSS?", k=2):
        print(f"[{doc.metadata['chunk_type']}] {doc.page_content[:200]}...\n")

    print("=== Telugu (script) ===")
    for doc in retrieve("నేర శాస్త్రం అంటే ఏమిటి?", k=2):
        print(f"[{doc.metadata['chunk_type']}] {doc.page_content[:200]}...\n")

    print("=== Tenglish ===")
    for doc in retrieve("criminology ante enti?", k=2):
        print(f"[{doc.metadata['chunk_type']}] {doc.page_content[:200]}...\n")