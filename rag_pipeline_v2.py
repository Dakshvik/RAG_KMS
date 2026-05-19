import json
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import ollama
import fasttext
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    SparseVector,
)
from sentence_transformers import CrossEncoder
from fastembed import SparseTextEmbedding

# ---------- CONFIGURATION ----------
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "bilingual_hybrid"          # your hybrid collection
PARENTS_FILE = "parent_pages/all_parents.json"
EMBED_MODEL = "bge-m3"                       # dense embeddings (Ollama)
FASTTEXT_MODEL_PATH = "lang_detect_model/lid.176.bin"
SPARSE_MODEL = "Qdrant/bm25"                 # same as in embed_index.py

# LLM models (Ollama tags) – quantized for 8 GB VRAM
LLM_FAST = "qwen3:1.7b"       # for rewriting, compression
LLM_HEAVY = "qwen3:4b"         # main generation

# Reranker (runs on CPU / GPU automatically)
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Tenglish keywords
TELUGU_LATIN_WORDS = {
    "ante", "enti", "ela", "unnav", "bagunnara", "emi", "enduku",
    "ekkada", "chesav", "vellava", "naaku", "neeku", "telugu",
    "cheppu", "ardham", "kadu", "avunu", "ledu", "vasthundi",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HybridRAG")
# -----------------------------------

# Load fastText once
ft_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

# Load parent documents
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


class HybridRAGPipeline:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        self.sparse_model = SparseTextEmbedding(SPARSE_MODEL)
        logger.info("Hybrid RAG Pipeline initialized.")

    # ---------- Language Detection ----------
    def detect_language(self, query: str) -> str:
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

    # ---------- Query Rewriting ----------
    def rewrite_query(self, query: str, lang: str) -> List[str]:
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
            logger.warning(f"Rewrite failed: {e}")
        return [query]

    # ---------- Hybrid Retrieval ----------
    def hybrid_retrieve(self, query: str, language_filter: Optional[str] = None, limit: int = 20) -> List[Any]:
        # 1. Dense query vector
        try:
            emb = ollama.embed(model=EMBED_MODEL, input=[query])
            dense_vec = emb["embeddings"][0]
        except Exception as e:
            logger.error(f"Dense embedding failed: {e}")
            return []

        # 2. Sparse query vector
        sparse_vec = None
        try:
            sparse_emb = list(self.sparse_model.embed([query]))[0]
            sparse_vec = SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            )
        except Exception as e:
            logger.warning(f"Sparse embedding failed, using dense only: {e}")

        # 3. Optional language filter
        qdrant_filter = None
        if language_filter:
            qdrant_filter = Filter(
                must=[FieldCondition(key="language", match=MatchValue(value=language_filter))]
            )

        # 4. Dense search – pass vector directly + using="dense"
        dense_response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vec,
            using="dense",
            query_filter=qdrant_filter,
            limit=limit * 2,
            with_payload=True,
        )
        dense_results = dense_response.points

        # 5. Sparse search – pass sparse vector directly + using="sparse"
        sparse_results = []
        if sparse_vec is not None:
            sparse_response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=sparse_vec,
                using="sparse",
                query_filter=qdrant_filter,
                limit=limit * 2,
                with_payload=True,
            )
            sparse_results = sparse_response.points

        # 6. Merge via Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        scores = {}
        for rank, point in enumerate(dense_results, start=1):
            scores[point.id] = scores.get(point.id, 0) + 1 / (rank + rrf_k)
        for rank, point in enumerate(sparse_results, start=1):
            scores[point.id] = scores.get(point.id, 0) + 1 / (rank + rrf_k)

        # 7. Sort, deduplicate, and return top-k
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:limit]
        id_to_point = {p.id: p for p in dense_results + sparse_results}
        merged = [id_to_point[pid] for pid in sorted_ids if pid in id_to_point]
        return merged

    # ---------- Reranking ----------
    def rerank_chunks(self, query: str, points: List[Any], top_k: int = 5) -> List[Any]:
        if not points:
            return []
        texts = [p.payload.get("text", "") for p in points]
        cross_input = [[query, t] for t in texts]
        scores = self.reranker.predict(cross_input)
        scored = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    # ---------- Context Assembly (Parent Pages) ----------
    def build_context(self, points: List[Any]) -> str:
        parent_keys = {}
        for p in points:
            doc_id = p.payload.get("doc_id")
            page = p.payload.get("page")
            if doc_id is not None and page is not None:
                parent_keys[f"{doc_id}::page{page}"] = None

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
        if len(context) <= 2000:
            return context
        system = (
            f"You are a context compressor. Given a user query and a large document, "
            f"extract ONLY the sentences and facts directly relevant to the query. "
            f"Discard everything else. Output the compressed text in {lang}."
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
            content = resp["message"]["content"].strip()
            if not content:
                logger.warning("Compression returned empty string, using original context.")
                return context
            return content
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return context

    # ---------- Dynamic Prompting & Generation ----------
    def detect_query_type(self, query: str) -> str:
        query_lower = query.lower()
        explanatory_keywords = [
            "explain", "how", "why", "describe", "elaborate", "compare", "contrast",
            "difference", "similarities", "analyse", "discuss",
            "వివరించు", "ఎలా", "ఎందుకు", "తేడా", "పోలిక",
        ]
        if any(kw in query_lower for kw in explanatory_keywords):
            return "explanatory"
        return "factual"

    def generate_answer(self, query: str, context: str, lang: str, query_type: str) -> str:
        if not context.strip():
            return (
                "క్షమించండి, సమాచారం లేదు." if lang == "Telugu"
                else "Sorry, no information found."
            )

        # Set temperature and max tokens based on query type
        if query_type == "factual":
            temperature = 0.1
            num_predict = 512
        else:  # explanatory
            temperature = 0.2
            num_predict = 1024

        # Try generation in requested language
        answer = self._call_ollama_generate(
            query, context, lang, query_type, temperature, num_predict
        )

        # If Telugu and empty, fallback to English
        if lang == "Telugu" and not answer:
            logger.warning("Telugu generation failed (empty response). Falling back to English.")
            answer = self._call_ollama_generate(
                query, context, "English", query_type, temperature, num_predict
            )
            if answer:
                answer = (
                    "⚠️ Telugu generation not available. Here is the answer in English:\n\n"
                    + answer
                )

        # Final fallback if still empty
        if not answer:
            logger.error("Generation completely failed after fallback.")
            return (
                "క్షమించండి, సమాచారాన్ని రూపొందించడంలో సమస్య ఏర్పడింది."
                if lang == "Telugu"
                else "Sorry, there was an issue generating the answer."
            )

        return answer

    def _call_ollama_generate(self, query, context, lang, query_type, temperature, num_predict):
        """Helper to avoid code duplication."""
        if query_type == "factual":
            system_prompt = (
                f"You are an expert knowledge assistant. Answer the user's question using ONLY the provided CONTEXT. "
                f"Be brief, precise, and highly factual. Extract the exact answer without adding any outside knowledge. "
                f"Respond entirely in {lang}. "
                f"CRITICAL: Do not write out internal thoughts, step-by-step analysis, or drafting processes. Output the final answer immediately."
            )
        else:
            system_prompt = (
                f"You are an expert knowledge assistant. Explain the concepts asked by the user clearly, "
                f"using ONLY the provided CONTEXT. Synthesize the steps or concepts logically to help the user understand. "
                f"Respond entirely in {lang}. Do not hallucinate or use outside knowledge. "
                f"CRITICAL: Do not write out internal thoughts, step-by-step analysis, or drafting processes. Output the final answer immediately."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"},
        ]
        try:
            resp = ollama.chat(
                model=LLM_HEAVY,
                messages=messages,
                options={"temperature": temperature, "num_predict": num_predict},
            )
            raw = resp["message"]["content"].strip()
            logger.info(f"Raw LLM response length ({lang}): {len(raw)} characters")
            return raw
        except Exception as e:
            logger.error(f"Generation call failed for {lang}: {e}")
            return ""

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

        # 3. Query rewriting (explanatory queries only)
        t0 = time.perf_counter()
        search_queries = [query]
        if any(kw in query.lower() for kw in ["explain", "how", "why", "describe", "compare", "వివరించు"]):
            search_queries = self.rewrite_query(query, lang)
        timings["rewriting"] = round(time.perf_counter() - t0, 3)

        # 4. Hybrid retrieval with language filter
        t0 = time.perf_counter()
        all_points = []
        for sq in search_queries:
            points = self.hybrid_retrieve(sq, language_filter=lang_filter, limit=30)
            all_points.extend(points)
        # Deduplicate by ID
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
                "routed_as": "RAG (no results)",
                "timings": timings,
            }

        # 5. Reranking
        t0 = time.perf_counter()
        top_points = self.rerank_chunks(query, all_points, top_k=5)
        timings["reranking"] = round(time.perf_counter() - t0, 3)

        # 6. Context assembly (parent pages)
        context = self.build_context(top_points)

        # 7. Compression (if needed)
        t0 = time.perf_counter()
        if len(context) > 4000:
            context = self.compress_context(query, context, lang)
        timings["compression"] = round(time.perf_counter() - t0, 3)

        # 8. Generate answer with dynamic prompt
        t0 = time.perf_counter()
        query_type = self.detect_query_type(query)
        answer = self.generate_answer(query, context, lang, query_type)
        timings["generation"] = round(time.perf_counter() - t0, 3)

        # 9. Citations
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
    pipeline = HybridRAGPipeline()
    test_queries = [
        "What is criminology?",
        "Compare and contrast IPC and BNSS",
        "నేర శాస్త్రం అంటే ఏమిటి?",
        "criminology ante enti?",
    ]
    for q in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {q}")
        result = pipeline.process_query(q)
        print(f"ROUTE: {result['routed_as']}")
        print(f"ANSWER:\n{result['answer']}")
        if result['citations']:
            print("CITATIONS:")
            for cit in result['citations']:
                print(f"  - {cit['file']} (page {cit['page']})")
        print("TIMINGS:", result['timings'])