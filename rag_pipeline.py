import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama
import fasttext
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder

# ---------- CONFIGURATION ----------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bilingualbilingual_hybrid_final_vectors"                # your Qdrant collection
PARENTS_FILE = "parent_pages/all_parents.json"
EMBED_MODEL = "bge-m3"                              # dense embeddings (Ollama)
FASTTEXT_MODEL_PATH = "lang_detect_model/lid.176.bin"

# LLM models (Ollama tags) – quantized versions mandatory for 8 GB VRAM
LLM_FAST = "qwen3:1.7b"       # for rewriting, compression (fast & cheap)
LLM_HEAVY = "qwen3:4b"         # main answer generation (quantized, e.g., q4_K_M)

# Reranker (runs on CPU / GPU automatically)
RERANKER_MODEL = "BAAI/bge-reranker-base"           # ~1.2 GB, excellent quality

# Tenglish keywords (same as in your retrieve.py)
TELUGU_LATIN_WORDS = {
    "ante", "enti", "ela", "unnav", "bagunnara", "emi", "enduku",
    "ekkada", "chesav", "vellava", "naaku", "neeku", "telugu",
    "cheppu", "ardham", "kadu", "avunu", "ledu", "vasthundi",
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("LocalRAG")
# -----------------------------------

# Load fastText model once
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# Load parent documents (one full page per key)
def load_parents(parents_file: str) -> Dict[str, str]:
    with open(parents_file, "r", encoding="utf-8") as f:
        parents_list = json.load(f)
    parents_dict = {}
    for p in parents_list:
        doc_id = p["metadata"]["doc_id"]
        page = p["metadata"]["page"]
        first_page = int(str(page).split("-")[0])
        parents_dict[f"{doc_id}::page{first_page}"] = p["page_content"]
    return parents_dict

parents_dict = load_parents(PARENTS_FILE)


class OptimizedRAGPipeline:
    def __init__(self):
        logger.info("Initializing Optimized RAG Pipeline (local qwen3:4b)...")
        self.client = QdrantClient(url=QDRANT_URL)
        # Reranker
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("Pipeline ready (Qdrant, Reranker, Ollama via calls).")

    # ---------- Language Detection ----------
    def detect_language(self, query: str) -> str:
        """Return 'Telugu' or 'English'."""
        if re.search(r'[\u0C00-\u0C7F]', query):
            return "Telugu"
        tokens = set(re.findall(r'\b\w+\b', query.lower()))
        if tokens & TELUGU_LATIN_WORDS:
            return "Telugu"
        label, _ = ft_model.predict(query.lower().strip(), k=1)
        lang = label[0].replace('__label__', '')
        return "Telugu" if lang == 'te' else "English"

    # ---------- Routing ----------
    def is_general_query(self, query: str) -> bool:
        """Small talk vs legal query."""
        q = query.lower().strip()
        general_patterns = [
            r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bthanks\b", r"\bthank you\b",
            r"\bbye\b", r"\bhow are you\b", r"\bwhat's up\b", r"\bwho are you\b",
            "హాయ్", "నమస్కారం", "థ్యాంక్", "ధన్యవాదాలు",
        ]
        legal_patterns = [
            r"\bsection\b", r"\blaw\b", r"\bcrime\b", r"\bcriminal\b", r"\barrest\b",
            r"\bpolice\b", r"\bfir\b", r"\bevidence\b", r"\bforensic\b",
            "సెక్షన్", "చట్టం", "పోలీస్"
        ]
        if any(re.search(p, q) for p in legal_patterns):
            return False
        if any(re.search(p, q) for p in general_patterns) and len(q) < 40:
            return True
        return False

    # ---------- Query Rewriting (Hypothetical Answer) ----------
    def rewrite_query(self, query: str, lang: str) -> List[str]:
        """
        Use the fast LLM to generate a hypothetical answer.
        This expanded text often improves retrieval recall.
        Returns [hypothetical_answer, original_query].
        """
        system = (
            f"You are a search assistant. Briefly answer the user's question in 2-3 sentences "
            f"in {lang} to help retrieve relevant documents. Do not add greetings."
        )
        try:
            resp = ollama.chat(
                model=LLM_FAST,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                options={"temperature": 0.1, "num_predict": 150},
            )
            content = resp["message"]["content"]
            if content and content.strip():
                return [content.strip(), query]
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
        return [query]

    # ---------- Dense Retrieval ----------
    def dense_retrieve(
        self, query: str, language_filter: Optional[str] = None, limit: int = 20
    ) -> List[Any]:
        """Embed with BGE-M3, query Qdrant, optionally filter by language."""
        # 1. Embed query
        try:
            emb = ollama.embed(model=EMBED_MODEL, input=[query])
            query_vec = emb["embeddings"][0]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

        # 2. Language filter (optional)
        qfilter = None
        if language_filter:
            qfilter = Filter(
                must=[FieldCondition(key="language", match=MatchValue(value=language_filter))]
            )

        # 3. Vector search
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
        return response.points

    # ---------- Reranking ----------
    def rerank_chunks(self, query: str, points: List[Any], top_k: int = 5) -> List[Any]:
        """Score candidates with cross-encoder, return top_k."""
        if not points:
            return []
        texts = [p.payload.get("text", "") for p in points]
        cross_input = [[query, t] for t in texts]
        scores = self.reranker.predict(cross_input)
        scored = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    # ---------- Context Assembly (Parent Pages) ----------
    def build_context(self, points: List[Any]) -> str:
        """
        From a list of child points, collect the unique parent pages.
        Sorted by page number for readability.
        """
        parent_keys = {}
        for p in points:
            doc_id = p.payload.get("doc_id")
            page = p.payload.get("page")
            if doc_id is not None and page is not None:
                parent_keys[f"{doc_id}::page{page}"] = None

        # Fetch parent texts, deduplicate
        seen = set()
        context_parts = []
        for key in parent_keys:
            text = parents_dict.get(key)
            if text and text not in seen:
                seen.add(text)
                context_parts.append(text)
        if context_parts:
            return "\n\n===== PAGE BREAK =====\n\n".join(context_parts)
        # Fallback to raw child texts
        return "\n\n".join([p.payload.get("text", "") for p in points[:5]])

    # ---------- Context Compression ----------
    def compress_context(self, query: str, context: str, lang: str) -> str:
        """Use fast LLM to extract only the sentences directly relevant to the query."""
        if len(context) <= 2000:
            return context  # don't compress small context
        system = (
            f"You are a context compressor. Given a user query and a large document, "
            f"extract ONLY the sentences and facts directly relevant to the query. "
            f"Discard everything else. Output the compressed text in {lang}. "
            f"Keep all important details, but remove irrelevant sections."
        )
        try:
            resp = ollama.chat(
                model=LLM_FAST,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"QUERY: {query}\n\nCONTEXT:\n{context}"},
                ],
                options={"temperature": 0.0, "num_predict": 1000},
            )
            content = resp["message"]["content"]
            return content.strip() if content else context
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return context

    # ---------- Final Generation ----------
    def generate_answer(self, query: str, context: str, lang: str) -> str:
        """Generate final answer with qwen3:4b, grounded strictly in context."""
        if not context.strip():
            return "క్షమించండి, సమాచారం లేదు." if lang == "Telugu" else "Sorry, no information found."

        system = (
            f"You are a legal assistant for Indian police. Answer the user's question "
            f"ONLY using the provided document context below. Be precise and factual. "
            f"Respond entirely in {lang}. Do not add any information not present in the context. "
            f"If the answer cannot be found, say so clearly."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"},
        ]
        try:
            resp = ollama.chat(
                model=LLM_HEAVY,
                messages=messages,
                options={"temperature": 0.1},
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Failed to generate an answer."

    # ---------- Main Orchestration ----------
    def process_query(self, query: str) -> Dict[str, Any]:
        start_total = time.perf_counter()
        timings = {}

        # 1. Language detection
        t0 = time.perf_counter()
        lang = self.detect_language(query)
        lang_filter = "english" if lang == "English" else "telugu"
        timings["language_detection"] = round(time.perf_counter() - t0, 3)
        logger.info(f"Language: {lang}")

        # 2. General chat shortcut
        if self.is_general_query(query):
            logger.info("Routing to General Chat")
            t0 = time.perf_counter()
            try:
                resp = ollama.chat(
                    model=LLM_HEAVY,
                    messages=[
                        {"role": "system", "content": f"Be a friendly assistant. Reply in {lang}."},
                        {"role": "user", "content": query},
                    ],
                    options={"temperature": 0.7},
                )
                answer = resp["message"]["content"].strip()
            except Exception:
                answer = "Hello! How can I help you today?"
            timings["generation"] = round(time.perf_counter() - t0, 3)
            timings["total"] = round(time.perf_counter() - start_total, 3)
            return {
                "query": query,
                "answer": answer,
                "citations": [],
                "routed_as": "General Chat",
                "timings": timings,
            }

        # 3. Query rewriting (optional, for explanatory queries)
        t0 = time.perf_counter()
        search_queries = [query]
        if any(kw in query.lower() for kw in ["explain", "how", "why", "describe", "compare", "వివరించు"]):
            search_queries = self.rewrite_query(query, lang)
        timings["rewriting"] = round(time.perf_counter() - t0, 3)

        # 4. Retrieval (dense, with language filter)
        t0 = time.perf_counter()
        all_points = []
        for sq in search_queries:
            points = self.dense_retrieve(sq, language_filter=lang_filter, limit=20)
            all_points.extend(points)
        # Deduplicate by point ID
        unique = {}
        for p in all_points:
            unique[p.id] = p
        all_points = list(unique.values())
        timings["retrieval"] = round(time.perf_counter() - t0, 3)

        if not all_points:
            timings["total"] = round(time.perf_counter() - start_total, 3)
            return {
                "query": query,
                "answer": "No relevant documents found.",
                "citations": [],
                "timings": timings,
            }

        # 5. Reranking
        t0 = time.perf_counter()
        top_points = self.rerank_chunks(query, all_points, top_k=5)
        timings["reranking"] = round(time.perf_counter() - t0, 3)

        # 6. Context assembly (parent pages)
        context = self.build_context(top_points)

        # 7. Compression (if context too long)
        t0 = time.perf_counter()
        if len(context) > 4000:
            context = self.compress_context(query, context, lang)
        timings["compression"] = round(time.perf_counter() - t0, 3)

        # 8. Generate answer
        t0 = time.perf_counter()
        answer = self.generate_answer(query, context, lang)
        timings["generation"] = round(time.perf_counter() - t0, 3)

        # 9. Build citations
        citations = []
        for p in top_points:
            payload = p.payload
            citations.append({
                "file": payload.get("filename", "Unknown"),
                "page": payload.get("page", "?"),
                "text": payload.get("text", "")[:200],
            })

        timings["total"] = round(time.perf_counter() - start_total, 3)
        return {
            "query": query,
            "answer": answer,
            "citations": citations,
            "routed_as": "RAG",
            "timings": timings,
        }


if __name__ == "__main__":
    pipeline = OptimizedRAGPipeline()
    test_queries = [
        "What is criminology?",
        # "compare and contrast IPC and BNSS"
        # "నేర శాస్త్రం అంటే ఏమిటి?",   # Telugu
        # "criminology ante enti?",         # Tenglish
    ]
    for q in test_queries:
        print("\n" + "="*60)
        print(f"QUERY: {q}")
        result = pipeline.process_query(q)
        print(f"ROUTE: {result['routed_as']}")
        print(f"ANSWER:\n{result['answer']}")
        if result['citations']:
            print("CITATIONS:")
            for cit in result['citations']:
                print(f"  - {cit['file']} (page {cit['page']})")
        print("TIMINGS:", result['timings'])