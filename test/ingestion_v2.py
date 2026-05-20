"""
Production PDF Extraction Pipeline
===================================
Dual-path extractor: Docling (fast, CPU/CUDA) + Marker (GPU vision fallback).
Handles CUDA OOM, corrupted fonts, multilingual PDFs, and incremental resumption.

Hardware target: NVIDIA RTX 4060 (8 GB VRAM) + 20-core CPU
"""

import gc
import os
import re
import json
import time
import logging
import unicodedata
import hashlib
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Environment config — set BEFORE any torch / model imports
# ---------------------------------------------------------------------------
os.environ.setdefault("TORCH_DEVICE", "cuda")
os.environ.setdefault("TORCH_DTYPE", "fp16")
# Prevent CUDA fragmentation on the RTX 4060's 8 GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
# Keep Surya/Texify batch sizes at 1 to stay within VRAM budget
os.environ.setdefault("SURYA_DET_BATCH_SIZE", "1")
os.environ.setdefault("SURYA_REC_BATCH_SIZE", "1")
os.environ.setdefault("TEXIFY_BATCH_SIZE", "1")

# Use physical core count; never over-subscribe
_cpu_cores = os.cpu_count() or 1
_thread_count = str(min(12, max(1, _cpu_cores - 2)))   # leave 2 cores for OS
os.environ.setdefault("OMP_NUM_THREADS", _thread_count)
os.environ.setdefault("MKL_NUM_THREADS", _thread_count)

# ---------------------------------------------------------------------------
# Deferred heavy imports (torch / docling / marker load slowly)
# ---------------------------------------------------------------------------
import torch

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PDFExtractor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_CHUNK_CHARS = 40          # discard near-empty chunks
CORRUPTION_THRESHOLD = 0.10   # stop-word ratio below this → likely garbled
STOP_WORDS = frozenset({"the", "be", "to", "of", "and", "a", "in", "that", "have", "it"})

LIGATURE_MAP = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00ad": "", "\u00a0": " ",
})

HEADERS_TO_SPLIT = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]

# ---------------------------------------------------------------------------
# ID utilities
# ---------------------------------------------------------------------------

def generate_doc_id(pdf_path: Path) -> str:
    subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
    key = f"{subject}::{pdf_path.name}".lower()
    return hashlib.md5(key.encode()).hexdigest()[:12]


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}-{chunk_index:04d}"

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_pdf_text(text: str, language: str = "english") -> str:
    """Normalise extracted PDF text; strips artifacts without losing content."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\uE000-\uF8FF]", "", text)          # private-use Unicode
    text = text.translate(LIGATURE_MAP)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)        # de-hyphenate line breaks
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)  # standalone page numbers
    if language == "english":
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_text_corrupted(text: str, threshold: float = CORRUPTION_THRESHOLD) -> bool:
    """Return True when stop-word density is suspiciously low (garbled font)."""
    if not text or len(text.strip()) < 50:
        return False
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    if len(words) < 15:
        return False
    ratio = sum(1 for w in words if w in STOP_WORDS) / len(words)
    return ratio < threshold

# ---------------------------------------------------------------------------
# CUDA memory management
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    return torch.cuda.is_available()


def free_cuda_memory() -> None:
    """Aggressively release CUDA memory after each heavy operation."""
    gc.collect()
    if cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def cuda_memory_guard(label: str = ""):
    """
    Context manager that frees CUDA memory on exit and converts OOM errors
    into a recoverable CudaOOMError so the caller can fall back gracefully.
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:
        free_cuda_memory()
        raise CudaOOMError(f"CUDA OOM during '{label}'") from exc
    finally:
        free_cuda_memory()


class CudaOOMError(RuntimeError):
    """Raised when a CUDA out-of-memory condition is detected."""


# ---------------------------------------------------------------------------
# Model manager — lazy singleton, supports explicit teardown
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Owns Docling and Marker model instances.
    Call teardown() to release VRAM between large batches if needed.
    """

    def __init__(self) -> None:
        self._docling_converter: Optional[DocumentConverter] = None
        self._docling_chunker: Optional[HierarchicalChunker] = None
        self._marker_converter: Optional[PdfConverter] = None
        self._marker_artifacts: Optional[Dict] = None
        self._md_splitter = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT)

    # -- Docling ------------------------------------------------------------

    @property
    def docling_converter(self) -> DocumentConverter:
        if self._docling_converter is None:
            logger.info("Loading Docling converter…")
            self._docling_converter = DocumentConverter()
        return self._docling_converter

    @property
    def docling_chunker(self) -> HierarchicalChunker:
        if self._docling_chunker is None:
            self._docling_chunker = HierarchicalChunker()
        return self._docling_chunker

    # -- Marker -------------------------------------------------------------

    @property
    def marker_converter(self) -> PdfConverter:
        if self._marker_converter is None:
            logger.info("Loading Marker models (CUDA)…")
            with cuda_memory_guard("marker_model_load"):
                self._marker_artifacts = create_model_dict()
                self._marker_converter = PdfConverter(artifact_dict=self._marker_artifacts)
        return self._marker_converter

    # -- Splitter -----------------------------------------------------------

    @property
    def md_splitter(self) -> MarkdownHeaderTextSplitter:
        return self._md_splitter

    # -- Lifecycle ----------------------------------------------------------

    def teardown_marker(self) -> None:
        """Release Marker VRAM.  Models are re-created on next access."""
        logger.info("Releasing Marker models from VRAM…")
        self._marker_converter = None
        self._marker_artifacts = None
        free_cuda_memory()

    def teardown_all(self) -> None:
        self._docling_converter = None
        self._docling_chunker = None
        self.teardown_marker()


# ---------------------------------------------------------------------------
# Document serialisation helper
# ---------------------------------------------------------------------------

def doc_to_dict(doc: Document) -> Dict[str, Any]:
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def dict_to_doc(d: Dict[str, Any]) -> Document:
    return Document(page_content=d["page_content"], metadata=d["metadata"])

# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class ProductionPDFExtractor:
    """
    Thread-unsafe but process-safe PDF extractor.
    Instantiate once and reuse across the whole batch.
    """

    def __init__(self) -> None:
        self._models = ModelManager()

    # -----------------------------------------------------------------------
    # Fast path — Docling (CPU/CUDA auto; excellent for digital PDFs)
    # -----------------------------------------------------------------------

    def _fast_path(self, pdf_path: Path) -> List[Document]:
        logger.info("⚡ Fast path (Docling): %s", pdf_path.name)
        try:
            result = self._models.docling_converter.convert(str(pdf_path))
        except Exception as exc:
            logger.warning("Docling conversion failed (%s): %s", pdf_path.name, exc)
            raise

        doc = result.document
        chunks = list(self._models.docling_chunker.chunk(doc))

        if not chunks:
            logger.warning("Docling produced 0 chunks for %s", pdf_path.name)
            return []

        # Corruption check on first 5 chunks
        sample = " ".join(c.text for c in chunks[:5])
        if is_text_corrupted(sample):
            logger.warning("Garbled font detected in %s → rerouting to Marker", pdf_path.name)
            return self._slow_path(pdf_path, language="english", method="marker_fallback")

        doc_id = generate_doc_id(pdf_path)
        documents: List[Document] = []

        for idx, chunk in enumerate(chunks):
            clean = clean_pdf_text(chunk.text, language="english")
            if len(clean) < MIN_CHUNK_CHARS:
                continue

            meta = chunk.meta
            heading = (meta.headings[0] if meta and meta.headings else "Unknown")

            page_no: Any = 1
            try:
                page_no = meta.doc_items[0].prov[0].page_no
            except (AttributeError, IndexError, TypeError):
                pass

            documents.append(Document(
                page_content=clean,
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": generate_chunk_id(doc_id, idx),
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "subject": pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem,
                    "language": "english",
                    "extraction_method": "docling_fast",
                    "heading": heading,
                    "page": page_no,
                },
            ))

        return documents

    # -----------------------------------------------------------------------
    # Slow path — Marker (GPU vision; handles scanned / multilingual PDFs)
    # -----------------------------------------------------------------------

    def _slow_path(
        self,
        pdf_path: Path,
        language: str,
        method: str,
        retry: int = 1,
    ) -> List[Document]:
        logger.info("🏎️  GPU path (Marker): %s [lang=%s]", pdf_path.name, language)

        for attempt in range(1, retry + 2):   # 1 normal + `retry` retries
            try:
                with cuda_memory_guard(f"marker:{pdf_path.name}"):
                    rendered = self._models.marker_converter(str(pdf_path))
                    # text_from_rendered returns (text, images, metadata) in ≥1.10
                    result = text_from_rendered(rendered)
                    text = result[0] if isinstance(result, (list, tuple)) else result
            except CudaOOMError:
                logger.error(
                    "CUDA OOM on attempt %d/%d for %s. Tearing down Marker and retrying…",
                    attempt, retry + 1, pdf_path.name,
                )
                self._models.teardown_marker()
                if attempt > retry:
                    logger.error("All retries exhausted for %s — skipping.", pdf_path.name)
                    return []
                time.sleep(2)
                continue
            except Exception as exc:
                logger.error("Marker failed on %s: %s", pdf_path.name, exc)
                return []
            break   # success

        if not text or not text.strip():
            logger.warning("Marker returned empty text for %s", pdf_path.name)
            return []

        md_docs = self._models.md_splitter.split_text(text)
        doc_id = generate_doc_id(pdf_path)
        documents: List[Document] = []

        for idx, md_doc in enumerate(md_docs):
            clean = clean_pdf_text(md_doc.page_content, language=language)
            if len(clean) < MIN_CHUNK_CHARS:
                continue

            heading = (
                md_doc.metadata.get("Header 3")
                or md_doc.metadata.get("Header 2")
                or md_doc.metadata.get("Header 1")
                or "Unknown"
            )

            documents.append(Document(
                page_content=clean,
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": generate_chunk_id(doc_id, idx),
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "subject": pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem,
                    "language": language,
                    "extraction_method": method,
                    "heading": heading,
                    "page": "vision_extracted",
                },
            ))

        return documents

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def extract_file(self, pdf_path_str: str, language: str = "english") -> List[Document]:
        """
        Extract a single PDF.  Returns an empty list (never raises) on failure
        so that batch processing can continue uninterrupted.
        """
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path_str}")

        lang = language.strip().lower()
        try:
            if lang in ("en", "english"):
                return self._fast_path(pdf_path)
            elif lang in ("te", "telugu"):
                return self._slow_path(pdf_path, language="telugu", method="marker_vision")
            else:
                logger.warning("Unknown language '%s' — defaulting to fast path.", language)
                return self._fast_path(pdf_path)
        except Exception as exc:
            logger.error(
                "Unhandled error extracting %s:\n%s",
                pdf_path.name,
                traceback.format_exc(),
            )
            return []

    def process_directory(
        self,
        data_dir: str,
        output_dir: str = "extracted_chunks",
        *,
        force: bool = False,
        vram_relief_every: int = 20,
    ) -> Dict[str, Any]:
        """
        Walk *data_dir* recursively, extract all PDFs, and write one JSON
        file per PDF into *output_dir*.

        Parameters
        ----------
        data_dir        Root directory containing PDF files.
        output_dir      Destination for extracted JSON chunk files.
        force           Re-extract even if an output file already exists.
        vram_relief_every
                        Call teardown_marker() every N files to prevent
                        cumulative VRAM fragmentation on long runs.

        Returns
        -------
        A summary dict with counts for processed / skipped / failed files.
        """
        base_path = Path(data_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        all_pdfs = sorted(base_path.rglob("*.pdf"))
        total = len(all_pdfs)
        logger.info("Found %d PDF(s) under '%s'", total, data_dir)

        stats: Dict[str, int] = {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

        for file_idx, pdf_path in enumerate(all_pdfs, start=1):
            doc_id = generate_doc_id(pdf_path)
            save_file = out_path / f"{doc_id}_{pdf_path.stem}.json"

            # Incremental resume
            if not force and save_file.exists():
                logger.info("⏭️  [%d/%d] Skipping (already done): %s", file_idx, total, pdf_path.name)
                stats["skipped"] += 1
                continue

            detected_lang = _detect_language_from_path(pdf_path)
            logger.info("[%d/%d] Processing: %s (lang=%s)", file_idx, total, pdf_path.name, detected_lang)

            t0 = time.perf_counter()
            docs = self.extract_file(str(pdf_path), language=detected_lang)
            elapsed = time.perf_counter() - t0

            if not docs:
                logger.warning("❌ No chunks produced for %s", pdf_path.name)
                stats["failed"] += 1
                continue

            # Write atomically: temp file → rename
            tmp_file = save_file.with_suffix(".tmp")
            try:
                with open(tmp_file, "w", encoding="utf-8") as fh:
                    json.dump([doc_to_dict(d) for d in docs], fh, ensure_ascii=False, indent=2)
                tmp_file.replace(save_file)
            except OSError as exc:
                logger.error("Failed writing %s: %s", save_file, exc)
                tmp_file.unlink(missing_ok=True)
                stats["failed"] += 1
                continue

            stats["processed"] += 1
            stats["total_chunks"] += len(docs)
            logger.info(
                "✅ %s → %d chunks [%.1fs]", pdf_path.name, len(docs), elapsed
            )

            # Periodic VRAM relief
            if file_idx % vram_relief_every == 0:
                logger.info("🧹 Periodic VRAM relief after %d files…", file_idx)
                self._models.teardown_marker()

        logger.info(
            "Batch complete. processed=%d  skipped=%d  failed=%d  total_chunks=%d",
            stats["processed"], stats["skipped"], stats["failed"], stats["total_chunks"],
        )
        return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_language_from_path(pdf_path: Path) -> str:
    parts_lower = [p.lower() for p in pdf_path.parts]
    if "telugu" in parts_lower:
        return "telugu"
    if "english" in parts_lower:
        return "english"
    return "english"   # safe default


def load_chunks_from_json(json_path: str) -> List[Document]:
    """Reload previously saved chunks from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return [dict_to_doc(d) for d in data]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = ProductionPDFExtractor()
    summary = extractor.process_directory(
        data_dir="data",
        output_dir="extracted_chunks",
        force=False,
        vram_relief_every=20,
    )
    print(json.dumps(summary, indent=2))