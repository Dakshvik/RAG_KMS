# =========================================================
# HYBRID RAG PIPELINE - PRODUCTION MONOLITHIC VERSION
# =========================================================
# FEATURES:
#
# ✅ XML MASTER PROMPTS
# ✅ GENERAL / LEGAL ROUTING
# ✅ FACTUAL / EXPLANATORY ROUTING
# ✅ QUERY REWRITE
# ✅ HYBRID DENSE + SPARSE RETRIEVAL
# ✅ RERANKING
# ✅ PARENT PAGE EXPANSION
# ✅ SMART CONTEXT ASSEMBLY
# ✅ CONDITIONAL COMPRESSION
# ✅ qwen3:1.7b utility tasks
# ✅ qwen3:8b reasoning
# ✅ FULL LOGGING
# ✅ TIMINGS
# ✅ FAILURE SAFETY
# ✅ NO SILENT FAILURES
# =========================================================

import json
import re
import time
import logging
import traceback
import sys
from typing import List, Dict, Any, Optional

# =========================================================
# FORCE LOG FLUSH
# =========================================================
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "rag_debug.log",
            encoding="utf-8"
        )
    ]
)

logger = logging.getLogger("HybridRAG")

logger.info("=" * 80)
logger.info("PROGRAM START")
logger.info("=" * 80)

# =========================================================
# IMPORTS
# =========================================================
try:

    logger.info("Importing libraries...")

    import ollama
    import fasttext

    from sentence_transformers import (
        CrossEncoder
    )

    from fastembed import (
        SparseTextEmbedding
    )

    from qdrant_client import (
        QdrantClient
    )

    from qdrant_client.models import (
        Filter,
        FieldCondition,
        MatchValue,
        SparseVector,
    )

    logger.info(
        "All imports successful"
    )

except Exception:

    logger.error("IMPORT FAILURE")

    traceback.print_exc()

    raise

# =========================================================
# CONFIG
# =========================================================
QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "bilingual_hybrid_final"

PARENTS_FILE = "parent_pages/all_parents.json"

FASTTEXT_MODEL_PATH = "lang_detect_model/lid.176.bin"

EMBED_MODEL = "bge-m3"

SPARSE_MODEL = "Qdrant/bm25"

# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------
LLM_FAST = "qwen3:1.7b"

LLM_HEAVY = "qwen3:8b"

RERANKER_MODEL = "BAAI/bge-reranker-base"

# ---------------------------------------------------------
# RETRIEVAL SETTINGS
# ---------------------------------------------------------
RETRIEVAL_LIMIT = 10

TOP_K_RERANK = 4

MAX_CONTEXT_CHARS = 6000

# =========================================================
# LANGUAGE WORDS
# =========================================================
TELUGU_LATIN_WORDS = {

    "ante",
    "enti",
    "ela",
    "unnav",
    "bagunnara",
    "emi",
    "enduku",
    "ekkada",
    "chesav",
}

# =========================================================
# XML MASTER PROMPT
# =========================================================
MASTER_SYSTEM_PROMPT = """
<role>
Legal/Kanoon AI – specialised assistant for the Indian Police
</role>

<subjects>

1. APPM (Andhra Pradesh Police Manual)
2. BNS
3. BNSS
4. BSA
5. IPC
6. CrPC
7. Indian Evidence Act
8. Criminology
9. Forensics
10. Cyber Crime
11. Investigation Procedures
12. Police Administration
13. Public Order
14. Special Crimes

</subjects>

<rules>

- Answer ONLY from provided CONTEXT
- Never hallucinate
- If context insufficient say so
- Follow language instruction exactly
- Do not invent legal sections
- Be accurate and factual

</rules>
"""

# =========================================================
# FACTUAL PROMPT
# =========================================================
FACTUAL_PROMPT_TEMPLATE = """
<type>
factual
</type>

<instruction>

You are an expert legal knowledge assistant.

Answer ONLY using the provided CONTEXT.

Rules:

- Be highly factual
- Extract exact legal information
- Do not explain unnecessarily
- Be concise
- Do not hallucinate
- If answer not present say so

Respond entirely in {lang}

</instruction>
"""

# =========================================================
# EXPLANATORY PROMPT
# =========================================================
EXPLANATORY_PROMPT_TEMPLATE = """
<type>
explanatory
</type>

<instruction>

You are an expert legal knowledge assistant.

Explain concepts clearly using ONLY the provided CONTEXT.

Rules:

- Explain logically
- Use structured reasoning
- Compare concepts if needed
- Do not hallucinate
- Do not use outside knowledge

Respond entirely in {lang}

</instruction>
"""

# =========================================================
# QUERY REWRITE PROMPT
# =========================================================
REWRITE_PROMPT_TEMPLATE = """
<task>

Rewrite the user's query into a highly searchable legal query.

Rules:

- Include likely act names
- Include punishment details if relevant
- Include procedural terms if relevant
- Do NOT answer
- Output ONLY rewritten query

Respond in {lang}

</task>
"""

# =========================================================
# COMPRESSION PROMPT
# =========================================================
COMPRESSION_PROMPT_TEMPLATE = """
You are a legal context compressor.

Extract ONLY the information relevant to the query.

Discard irrelevant text.

Keep:
- legal sections
- punishments
- procedures
- definitions
- evidence details

Output concise relevant context only.

Respond in {lang}
"""

# =========================================================
# CLASS
# =========================================================
class HybridRAGPipeline:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self):

        logger.info("=" * 80)
        logger.info("PIPELINE INIT START")
        logger.info("=" * 80)

        try:

            # -------------------------------------------------
            # FASTTEXT
            # -------------------------------------------------
            logger.info(
                "Loading fasttext..."
            )

            self.ft_model = fasttext.load_model(
                FASTTEXT_MODEL_PATH
            )

            logger.info(
                "Fasttext loaded"
            )

            # -------------------------------------------------
            # RERANKER
            # -------------------------------------------------
            logger.info(
                "Loading reranker..."
            )

            self.reranker = CrossEncoder(
                RERANKER_MODEL,
                max_length=512
            )

            logger.info(
                "Reranker loaded"
            )

            # -------------------------------------------------
            # SPARSE MODEL
            # -------------------------------------------------
            logger.info(
                "Loading sparse model..."
            )

            self.sparse_model = SparseTextEmbedding(
                SPARSE_MODEL
            )

            logger.info(
                "Sparse model loaded"
            )

            # -------------------------------------------------
            # QDRANT
            # -------------------------------------------------
            logger.info(
                "Connecting to Qdrant..."
            )

            self.client = QdrantClient(
                url=QDRANT_URL,
                timeout=60,
            )

            logger.info(
                "Qdrant connected"
            )

            # -------------------------------------------------
            # COLLECTIONS
            # -------------------------------------------------
            collections = self.client.get_collections()

            logger.info(
                f"Collections: "
                f"{[c.name for c in collections.collections]}"
            )

            # -------------------------------------------------
            # PARENT PAGES
            # -------------------------------------------------
            logger.info(
                "Loading parent pages..."
            )

            with open(
                PARENTS_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                parents_list = json.load(f)

            self.parents_dict = {}

            for p in parents_list:

                try:

                    doc_id = p["metadata"]["doc_id"]

                    page = p["metadata"]["page"]

                    first_page = int(
                        str(page).split("-")[0]
                    )

                    key = (
                        f"{doc_id}::page{first_page}"
                    )

                    self.parents_dict[key] = (
                        p["page_content"]
                    )

                except Exception:

                    logger.warning(
                        "Failed parent page parse"
                    )

            logger.info(
                f"Loaded parents: "
                f"{len(self.parents_dict)}"
            )

            logger.info("=" * 80)
            logger.info("PIPELINE INIT SUCCESS")
            logger.info("=" * 80)

        except Exception:

            logger.error(
                "PIPELINE INIT FAILED"
            )

            traceback.print_exc()

            raise

    # =====================================================
    # SAFE OLLAMA CHAT
    # =====================================================
    def safe_chat(
        self,
        model,
        messages,
        options=None
    ):

        try:

            start = time.perf_counter()

            resp = ollama.chat(
                model=model,
                messages=messages,
                options=options or {}
            )

            elapsed = round(
                time.perf_counter() - start,
                2
            )

            logger.info(
                f"{model} completed "
                f"({elapsed}s)"
            )

            return resp["message"]["content"]

        except Exception:

            logger.error(
                f"Ollama chat failed "
                f"for {model}"
            )

            traceback.print_exc()

            return ""

    # =====================================================
    # LANGUAGE DETECTION
    # =====================================================
    def detect_language(
        self,
        query: str
    ):

        logger.info(
            "Detecting language..."
        )

        try:

            if re.search(
                r'[\u0C00-\u0C7F]',
                query
            ):
                return "Telugu"

            tokens = set(
                re.findall(
                    r'\b\w+\b',
                    query.lower()
                )
            )

            if tokens & TELUGU_LATIN_WORDS:
                return "Telugu"

            label, _ = self.ft_model.predict(
                query.lower().strip(),
                k=1
            )

            lang = label[0].replace(
                '__label__',
                ''
            )

            return (
                "Telugu"
                if lang == "te"
                else "English"
            )

        except Exception:

            traceback.print_exc()

            return "English"

    # =====================================================
    # GENERAL QUERY DETECTION
    # =====================================================
    def is_general_query(
        self,
        query: str
    ):

        q = query.lower().strip()

        general_patterns = [

            r"\bhi\b",
            r"\bhello\b",
            r"\bhey\b",
            r"\bthanks\b",
            r"\bthank you\b",
            r"\bbye\b",
            r"\bhow are you\b",
            r"\bwho are you\b",

            "హాయ్",
            "నమస్కారం",
        ]

        legal_patterns = [

            r"\bsection\b",
            r"\blaw\b",
            r"\bcrime\b",
            r"\bpolice\b",
            r"\bfir\b",
            r"\bipc\b",
            r"\bbns\b",
            r"\bbnss\b",
            r"\bbsa\b",
            r"\bdowry\b",
            r"\bbail\b",
            r"\bpunishment\b",
        ]

        if any(
            re.search(p, q)
            for p in legal_patterns
        ):
            return False

        if any(
            re.search(p, q)
            for p in general_patterns
        ) and len(q) < 40:
            return True

        return False

    # =====================================================
    # QUERY TYPE
    # =====================================================
    def detect_query_type(
        self,
        query
    ):

        query_lower = query.lower()

        explanatory_keywords = [

            "explain",
            "why",
            "how",
            "compare",
            "difference",
            "analyse",
            "discuss",

            "వివరించు",
            "ఎలా",
            "ఎందుకు",
        ]

        if any(
            kw in query_lower
            for kw in explanatory_keywords
        ):
            return "explanatory"

        return "factual"

    # =====================================================
    # QUERY REWRITE
    # =====================================================
    def rewrite_query(
        self,
        query,
        lang
    ):

        logger.info("=" * 80)
        logger.info("QUERY REWRITE")
        logger.info("=" * 80)

        try:

            system = (
                REWRITE_PROMPT_TEMPLATE.format(
                    lang=lang
                )
            )

            rewritten = self.safe_chat(

                model=LLM_FAST,

                messages=[

                    {
                        "role": "system",
                        "content": system
                    },

                    {
                        "role": "user",
                        "content": query
                    }
                ],

                options={
                    "temperature": 0.1,
                    "num_predict": 200,
                }
            )

            if rewritten.strip():

                logger.info(
                    f"Rewrite: {rewritten}"
                )

                return [
                    rewritten.strip(),
                    query
                ]

            return [query]

        except Exception:

            traceback.print_exc()

            return [query]

    # =====================================================
    # HYBRID RETRIEVAL
    # =====================================================
    def hybrid_retrieve(
        self,
        query,
        language_filter=None,
        limit=RETRIEVAL_LIMIT
    ):

        logger.info("=" * 80)
        logger.info("HYBRID RETRIEVAL")
        logger.info("=" * 80)

        try:

            # -------------------------------------------------
            # DENSE EMBEDDING
            # -------------------------------------------------
            emb = ollama.embed(
                model=EMBED_MODEL,
                input=[query]
            )

            dense_vec = emb["embeddings"][0]

            # -------------------------------------------------
            # SPARSE EMBEDDING
            # -------------------------------------------------
            sparse_emb = list(
                self.sparse_model.embed([query])
            )[0]

            sparse_vec = SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            )

            # -------------------------------------------------
            # FILTER
            # -------------------------------------------------
            qdrant_filter = None

            if language_filter:

                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="language",
                            match=MatchValue(
                                value=language_filter
                            )
                        )
                    ]
                )

            # -------------------------------------------------
            # DENSE SEARCH
            # -------------------------------------------------
            dense_response = (
                self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=dense_vec,
                    using="dense",
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                )
            )

            dense_results = (
                dense_response.points
            )

            # -------------------------------------------------
            # SPARSE SEARCH
            # -------------------------------------------------
            sparse_response = (
                self.client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=sparse_vec,
                    using="sparse",
                    query_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                )
            )

            sparse_results = (
                sparse_response.points
            )

            # -------------------------------------------------
            # RRF MERGE
            # -------------------------------------------------
            rrf_k = 60

            scores = {}

            for rank, point in enumerate(
                dense_results,
                start=1
            ):
                scores[point.id] = (
                    scores.get(point.id, 0)
                    + 1 / (rank + rrf_k)
                )

            for rank, point in enumerate(
                sparse_results,
                start=1
            ):
                scores[point.id] = (
                    scores.get(point.id, 0)
                    + 1 / (rank + rrf_k)
                )

            sorted_ids = sorted(
                scores,
                key=scores.get,
                reverse=True
            )

            id_to_point = {

                p.id: p
                for p in (
                    dense_results
                    + sparse_results
                )
            }

            merged = [

                id_to_point[pid]

                for pid in sorted_ids

                if pid in id_to_point
            ]

            logger.info(
                f"Retrieved "
                f"{len(merged)} points"
            )

            return merged

        except Exception:

            logger.error(
                "Retrieval failed"
            )

            traceback.print_exc()

            return []

    # =====================================================
    # RERANK
    # =====================================================
    def rerank_chunks(
        self,
        query,
        points,
        top_k=TOP_K_RERANK
    ):

        logger.info("=" * 80)
        logger.info("RERANKING")
        logger.info("=" * 80)

        try:

            if not points:
                return []

            texts = [

                p.payload.get("text", "")

                for p in points
            ]

            cross_input = [

                [query, t]

                for t in texts
            ]

            scores = self.reranker.predict(
                cross_input
            )

            scored = sorted(

                zip(scores, points),

                key=lambda x: x[0],

                reverse=True
            )

            return [

                p for _, p in scored[:top_k]
            ]

        except Exception:

            logger.error(
                "Rerank failed"
            )

            traceback.print_exc()

            return []

    # =====================================================
    # BUILD CONTEXT
    # =====================================================
    def build_context(
        self,
        points
    ):

        logger.info("=" * 80)
        logger.info("BUILD CONTEXT")
        logger.info("=" * 80)

        try:

            context_parts = []

            seen = set()

            for p in points:

                payload = p.payload

                doc_id = payload.get("doc_id")

                page = payload.get("page")

                text = payload.get("text", "")

                # -------------------------------------------------
                # PARENT PAGE
                # -------------------------------------------------
                parent_key = (
                    f"{doc_id}::page{page}"
                )

                parent_text = (
                    self.parents_dict.get(
                        parent_key
                    )
                )

                final_text = (
                    parent_text
                    if parent_text
                    else text
                )

                if (
                    final_text
                    and final_text not in seen
                ):

                    seen.add(final_text)

                    context_parts.append(
                        final_text
                    )

            context = (
                "\n\n===== PAGE BREAK =====\n\n"
                .join(context_parts)
            )

            logger.info(
                f"Context length: "
                f"{len(context)}"
            )

            return context

        except Exception:

            logger.error(
                "Context build failed"
            )

            traceback.print_exc()

            return ""

    # =====================================================
    # CONTEXT COMPRESSION
    # =====================================================
    def compress_context(
        self,
        query,
        context,
        lang
    ):

        logger.info("=" * 80)
        logger.info("CONTEXT COMPRESSION")
        logger.info("=" * 80)

        try:

            if len(context) <= MAX_CONTEXT_CHARS:

                logger.info(
                    "Compression skipped"
                )

                return context

            system = (
                COMPRESSION_PROMPT_TEMPLATE
                .format(lang=lang)
            )

            compressed = self.safe_chat(

                model=LLM_FAST,

                messages=[

                    {
                        "role": "system",
                        "content": system
                    },

                    {
                        "role": "user",
                        "content":
                        f"QUERY:\n{query}\n\n"
                        f"CONTEXT:\n{context}"
                    }
                ],

                options={
                    "temperature": 0.0,
                    "num_predict": 1200,
                }
            )

            if compressed.strip():

                logger.info(
                    f"Compressed from "
                    f"{len(context)} "
                    f"to "
                    f"{len(compressed)}"
                )

                return compressed

            return context

        except Exception:

            logger.error(
                "Compression failed"
            )

            traceback.print_exc()

            return context

    # =====================================================
    # FINAL GENERATION
    # =====================================================
    def generate_answer(
        self,
        query,
        context,
        lang,
        query_type
    ):

        logger.info("=" * 80)
        logger.info("FINAL GENERATION")
        logger.info("=" * 80)

        try:

            if not context.strip():

                return (
                    "No relevant information found."
                )

            # -------------------------------------------------
            # DYNAMIC PROMPT
            # -------------------------------------------------
            if query_type == "factual":

                dynamic_prompt = (
                    FACTUAL_PROMPT_TEMPLATE
                    .format(lang=lang)
                )

                temperature = 0.1

                num_predict = 600

            else:

                dynamic_prompt = (
                    EXPLANATORY_PROMPT_TEMPLATE
                    .format(lang=lang)
                )

                temperature = 0.2

                num_predict = 1200

            full_system_prompt = (
                MASTER_SYSTEM_PROMPT
                + "\n\n"
                + dynamic_prompt
            )

            answer = self.safe_chat(

                model=LLM_HEAVY,

                messages=[

                    {
                        "role": "system",
                        "content":
                        full_system_prompt
                    },

                    {
                        "role": "user",
                        "content":
                        f"CONTEXT:\n{context}\n\n"
                        f"QUESTION:\n{query}"
                    }
                ],

                options={
                    "temperature":
                    temperature,

                    "num_predict":
                    num_predict,
                }
            )

            return answer

        except Exception:

            logger.error(
                "Generation failed"
            )

            traceback.print_exc()

            return (
                "Generation failed."
            )

    # =====================================================
    # MAIN PIPELINE
    # =====================================================
    def process_query(
        self,
        query
    ):

        logger.info("=" * 80)
        logger.info(f"QUERY: {query}")
        logger.info("=" * 80)

        total_start = (
            time.perf_counter()
        )

        timings = {}

        try:

            # -------------------------------------------------
            # LANGUAGE
            # -------------------------------------------------
            t0 = time.perf_counter()

            lang = self.detect_language(
                query
            )

            lang_filter = (
                "english"
                if lang == "English"
                else "telugu"
            )

            timings["language"] = round(
                time.perf_counter() - t0,
                2
            )

            # -------------------------------------------------
            # GENERAL CHAT ROUTING
            # -------------------------------------------------
            if self.is_general_query(query):

                logger.info(
                    "ROUTING TO GENERAL CHAT"
                )

                answer = self.safe_chat(

                    model=LLM_FAST,

                    messages=[

                        {
                            "role": "system",
                            "content":
                            f"You are a friendly assistant. "
                            f"Reply in {lang}."
                        },

                        {
                            "role": "user",
                            "content": query
                        }
                    ],

                    options={
                        "temperature": 0.7,
                        "num_predict": 200,
                    }
                )

                return {

                    "query": query,

                    "answer": answer,

                    "route":
                    "general_chat",

                    "citations": [],

                    "timings": timings,
                }

            # -------------------------------------------------
            # QUERY TYPE
            # -------------------------------------------------
            query_type = (
                self.detect_query_type(
                    query
                )
            )

            logger.info(
                f"Query type: "
                f"{query_type}"
            )

            # -------------------------------------------------
            # REWRITE
            # -------------------------------------------------
            t0 = time.perf_counter()

            search_queries = (
                self.rewrite_query(
                    query,
                    lang
                )
            )

            timings["rewrite"] = round(
                time.perf_counter() - t0,
                2
            )

            # -------------------------------------------------
            # RETRIEVAL
            # -------------------------------------------------
            t0 = time.perf_counter()

            all_points = []

            for sq in search_queries:

                points = (
                    self.hybrid_retrieve(
                        sq,
                        language_filter=lang_filter,
                        limit=RETRIEVAL_LIMIT
                    )
                )

                all_points.extend(points)

            unique = {}

            for p in all_points:
                unique[p.id] = p

            all_points = list(
                unique.values()
            )

            timings["retrieval"] = round(
                time.perf_counter() - t0,
                2
            )

            if not all_points:

                return {

                    "query": query,

                    "answer":
                    "No relevant documents found.",

                    "route": "rag",

                    "citations": [],

                    "timings": timings,
                }

            # -------------------------------------------------
            # RERANK
            # -------------------------------------------------
            t0 = time.perf_counter()

            top_points = (
                self.rerank_chunks(
                    query,
                    all_points,
                    top_k=TOP_K_RERANK
                )
            )

            timings["rerank"] = round(
                time.perf_counter() - t0,
                2
            )

            # -------------------------------------------------
            # CONTEXT
            # -------------------------------------------------
            t0 = time.perf_counter()

            context = self.build_context(
                top_points
            )

            timings["context"] = round(
                time.perf_counter() - t0,
                2
            )

            # -------------------------------------------------
            # COMPRESSION
            # -------------------------------------------------
            t0 = time.perf_counter()

            context = self.compress_context(
                query,
                context,
                lang
            )

            timings["compression"] = round(
                time.perf_counter() - t0,
                2
            )

            # -------------------------------------------------
            # FINAL GENERATION
            # -------------------------------------------------
            t0 = time.perf_counter()

            answer = self.generate_answer(
                query,
                context,
                lang,
                query_type
            )

            timings["generation"] = round(
                time.perf_counter() - t0,
                2
            )

            # -------------------------------------------------
            # CITATIONS
            # -------------------------------------------------
            citations = []

            for p in top_points:

                payload = p.payload

                citations.append({

                    "file":
                    payload.get(
                        "filename",
                        "Unknown"
                    ),

                    "page":
                    payload.get(
                        "page",
                        "?"
                    ),

                    "text":
                    payload.get(
                        "text",
                        ""
                    )[:300]
                })

            timings["total"] = round(
                time.perf_counter()
                - total_start,
                2
            )

            logger.info("=" * 80)
            logger.info("PIPELINE SUCCESS")
            logger.info("=" * 80)

            return {

                "query": query,

                "answer": answer,

                "route": "rag",

                "query_type":
                query_type,

                "citations":
                citations,

                "timings":
                timings,
            }

        except Exception:

            logger.error(
                "PROCESS QUERY FAILED"
            )

            traceback.print_exc()

            return {
                "answer":
                "Pipeline failed."
            }

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    try:

        pipeline = HybridRAGPipeline()

        test_queries = [

            "Hi",

            "How much fine should we pay if anyone demands dowry as per dowry prohibition act?",

            "Explain difference between IPC and BNSS"
        ]

        for q in test_queries:

            print("\n" + "=" * 80)

            result = (
                pipeline.process_query(q)
            )

            print("\nFINAL ANSWER:\n")

            print(result["answer"])

            print("\nROUTE:")

            print(result["route"])

            print("\nTIMINGS:")

            print(result["timings"])

            print("\n" + "=" * 80)

    except Exception:

        logger.error(
            "FATAL PROGRAM FAILURE"
        )

        traceback.print_exc()