"""
Production PDF Extraction Pipeline v3 — Memory Thrashing Fix
==============================================================
Dual-path extractor: Docling (fast, CPU/CUDA) + Marker (GPU vision with streaming).
Handles CUDA OOM, corrupted fonts, multilingual PDFs, and incremental resumption.

KEY FIX (v3):
- Implements page-range streaming for Marker to prevent 2,536+ bbox accumulation
- Processes pages in configurable batches (default: 2 pages per batch)
- Recognizes text immediately after detection within each batch
- Flushes chunks to disk after each batch (prevents RAM accumulation)
- VRAM monitoring with auto-throttling (pauses/bails at high pressure)

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
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Environment config — set BEFORE any torch / model imports
# ---------------------------------------------------------------------------
os.environ.setdefault("TORCH_DEVICE", "cuda")
os.environ.setdefault("TORCH_DTYPE", "fp16")
# CRITICAL: Reduce split size for 8GB VRAM (was 128, now 64)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
# Force Docling to CPU to save VRAM
os.environ.setdefault("DOCLING_DEVICE", "cpu")
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

# MASSIVE VRAM SAVER: Disable gradients globally (no backprop needed)
torch.set_grad_enabled(False)

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

# Streaming parameters: CRITICAL for preventing memory thrash
MARKER_BATCH_PAGES = 4        # Process 4 pages at a time (optimal for 8GB VRAM with v4 fixes)
MARKER_BATCH_TIMEOUT = 300    # 5 min timeout per batch

# VRAM monitoring & auto-throttling (v3.1)
VRAM_THROTTLE_THRESHOLD = 15  # Pause if free VRAM drops below 15% (i.e., 85% used)
VRAM_THROTTLE_SLEEP = 3       # Sleep 3s to let memory settle
VRAM_CRITICAL_THRESHOLD = 5   # Emergency: bail if free drops below 5% (i.e., 95% used)

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


def hard_cuda_reset() -> None:
    """
    REAL CUDA cleanup - much stronger than empty_cache().
    Includes IPC cleanup and peak memory reset.
    """
    gc.collect()
    if cuda_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        torch.cuda.synchronize()


def get_vram_stats() -> Tuple[float, float, float]:
    """
    Get current VRAM status.
    Returns: (used_gb, total_gb, free_percent)
    """
    if not cuda_available():
        return (0.0, 0.0, 100.0)
    
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    used_gb = used_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    free_percent = (free_bytes / total_bytes) * 100
    
    return (used_gb, total_gb, free_percent)


def log_vram_status(label: str = "") -> None:
    """Log current VRAM usage for monitoring."""
    used_gb, total_gb, free_percent = get_vram_stats()
    status = f"VRAM: {used_gb:.2f}/{total_gb:.2f} GB ({free_percent:.1f}% free)"
    if label:
        logger.info(f"{label} — {status}")
    else:
        logger.info(status)


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
# PDF utilities — page extraction for streaming
# ---------------------------------------------------------------------------

def get_pdf_page_count(pdf_path: str) -> int:
    """
    Fast query: How many pages does this PDF have?
    Uses pypdf to count pages without loading entire document.
    """
    try:
        import pypdf
        with open(pdf_path, "rb") as fh:
            reader = pypdf.PdfReader(fh)
            return len(reader.pages)
    except ImportError:
        logger.warning("pypdf not installed; falling back to Marker page detection")
        return None
    except Exception as exc:
        logger.warning("Could not detect page count for %s: %s", pdf_path, exc)
        return None


def extract_pdf_page_range(
    pdf_path: str,
    output_pdf: str,
    page_start: int = 0,
    page_end: Optional[int] = None,
) -> bool:
    """
    Extract pages [page_start:page_end] from PDF into a new PDF file.
    Returns True on success, False on failure.
    """
    try:
        import pypdf
        with open(pdf_path, "rb") as fh:
            reader = pypdf.PdfReader(fh)
            writer = pypdf.PdfWriter()
            
            total = len(reader.pages)
            end = min(page_end or total, total)
            
            for i in range(page_start, end):
                writer.add_page(reader.pages[i])
            
            with open(output_pdf, "wb") as out_fh:
                writer.write(out_fh)
            return True
    except Exception as exc:
        logger.error("Failed to extract page range from %s: %s", pdf_path, exc)
        return False


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

    @property
    def marker_converter(self) -> PdfConverter:
        if self._marker_converter is None:
            logger.info("Loading Marker models (CUDA)…")
            with cuda_memory_guard("marker_model_load"):
                self._marker_artifacts = create_model_dict()
                self._marker_converter = PdfConverter(artifact_dict=self._marker_artifacts)
        return self._marker_converter

    @property
    def md_splitter(self) -> MarkdownHeaderTextSplitter:
        return self._md_splitter

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
# Core extractor v3
# ---------------------------------------------------------------------------

class ProductionPDFExtractor:
    """
    Thread-unsafe but process-safe PDF extractor.
    v3.1 Features:
    - Streaming batches (2 pages at a time)
    - Flushes chunks to disk after each batch
    - VRAM monitoring with auto-throttling
    """

    def __init__(self, marker_batch_pages: int = MARKER_BATCH_PAGES) -> None:
        self._models = ModelManager()
        self._marker_batch_pages = marker_batch_pages

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
            logger.warning("Docling produced 0 chunks for %s. Falling back to Marker streaming…", pdf_path.name)
            return self._slow_path_streaming(pdf_path, language="english", method="marker_fallback")

        # Corruption check on first 5 chunks
        sample = " ".join(c.text for c in chunks[:5])
        if is_text_corrupted(sample):
            logger.warning("Garbled font detected in %s → rerouting to Marker", pdf_path.name)
            return self._slow_path_streaming(pdf_path, language="english", method="marker_fallback")

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
    # Slow path v3 — Marker with STREAMING (prevents bbox accumulation)
    # -----------------------------------------------------------------------

    def _slow_path_streaming(
        self,
        pdf_path: Path,
        language: str,
        method: str,
        output_file: Optional[str] = None,
    ):
        """
        Process PDF in page-range batches to prevent memory thrashing.
        Writes chunks directly to output_file (JSONL format) to prevent RAM accumulation.
        Returns: chunk count (int)
        """
        logger.info("🏎️  GPU path with Streaming (Marker): %s [lang=%s, batch=%d pages]",
                    pdf_path.name, language, self._marker_batch_pages)

        # Detect total page count
        total_pages = get_pdf_page_count(str(pdf_path))
        if total_pages is None:
            logger.warning("Could not detect page count; falling back to full-PDF processing")
            return self._slow_path_single(pdf_path, language, method)

        logger.info("Detected %d pages; will process in batches of %d", total_pages, self._marker_batch_pages)

        doc_id = generate_doc_id(pdf_path)
        chunk_counter = 0
        
        # Determine output file: use provided output_file or temp buffer
        if output_file:
            out_file_path = Path(output_file)
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            output_buffer = out_file_path
        else:
            output_buffer = Path(tempfile.gettempdir()) / f"{doc_id}_streaming.jsonl"

        # Process in page-range batches
        temp_dir = tempfile.gettempdir()
        for batch_idx, page_start in enumerate(range(0, total_pages, self._marker_batch_pages)):
            page_end = min(page_start + self._marker_batch_pages, total_pages)
            batch_pages = page_end - page_start

            # Check VRAM before starting batch
            used_gb, total_gb, free_pct = get_vram_stats()
            logger.info(
                "📄 Batch %d: Processing pages [%d:%d] (%d pages) [VRAM: %.1f/%.1f GB, %.1f%% free]",
                batch_idx + 1, page_start, page_end, batch_pages, used_gb, total_gb, free_pct
            )
            
            # SAFETY: If VRAM critically high, pause to let OS reclaim memory
            if free_pct < VRAM_THROTTLE_THRESHOLD:
                logger.warning(
                    "⚠️  VRAM pressure high (%.1f%% free, threshold: %.1f%%). "
                    "Pausing %.1fs to let OS reclaim memory…",
                    free_pct, VRAM_THROTTLE_THRESHOLD, VRAM_THROTTLE_SLEEP
                )
                time.sleep(VRAM_THROTTLE_SLEEP)
                free_cuda_memory()
                used_gb, total_gb, free_pct = get_vram_stats()
                logger.info("After pause: VRAM now %.1f/%.1f GB (%.1f%% free)", used_gb, total_gb, free_pct)
            
            # EMERGENCY: If still critical, skip this batch to prevent thrashing
            if free_pct < VRAM_CRITICAL_THRESHOLD:
                logger.error(
                    "🔴 CRITICAL VRAM (%.1f%% free). Skipping batch [%d:%d] to prevent thrashing.",
                    free_pct, page_start, page_end
                )
                continue

            # Extract this batch to a temporary PDF
            temp_pdf = Path(temp_dir) / f"marker_batch_{batch_idx}.pdf"
            if not extract_pdf_page_range(str(pdf_path), str(temp_pdf), page_start, page_end):
                logger.error("Failed to extract page batch [%d:%d]", page_start, page_end)
                temp_pdf.unlink(missing_ok=True)
                continue

            # Process this batch with Marker
            try:
                batch_docs = self._process_marker_batch(
                    temp_pdf, doc_id, chunk_counter, language, method
                )
                chunk_counter += len(batch_docs)
                
                # **FLUSH TO DISK IMMEDIATELY** — don't accumulate in RAM
                if batch_docs:
                    with open(output_buffer, "a", encoding="utf-8") as fh:
                        for doc in batch_docs:
                            fh.write(json.dumps(doc_to_dict(doc)) + "\n")
                    logger.info("📝 Flushed %d chunks to disk", len(batch_docs))
                    
                    # CLEANUP: Delete batch docs immediately after writing
                    del batch_docs
                    gc.collect()
                    
            except Exception as exc:
                logger.error("Marker batch [%d:%d] failed: %s", page_start, page_end, exc)
            finally:
                # CRITICAL: Clean up temp PDF immediately
                temp_pdf.unlink(missing_ok=True)
                free_cuda_memory()
                
                # Log final VRAM state of batch
                used_gb, total_gb, free_pct = get_vram_stats()
                logger.info(
                    "✅ Batch %d cleaned. VRAM: %.1f/%.1f GB (%.1f%% free)",
                    batch_idx + 1, used_gb, total_gb, free_pct
                )

        # Handle output based on whether we're streaming to a file or returning documents
        if output_file:
            # Case 1: Streaming to output_file (used for process_directory)
            # Count chunks from output file WITHOUT loading them into RAM
            chunk_count = 0
            try:
                with open(output_buffer, "r", encoding="utf-8") as fh:
                    chunk_count = sum(1 for line in fh if line.strip())
                logger.info("✅ Streaming complete: %d total chunks extracted (on disk at %s)", chunk_count, output_file)
            except FileNotFoundError:
                logger.warning("No buffer file found; streaming produced 0 chunks")
            # Keep file for process_directory to use
            return chunk_count
        else:
            # Case 2: Loading into memory for fallback (used from _fast_path)
            # Load all chunks from buffer into memory
            logger.info("📖 Loading chunks from disk buffer into memory…")
            documents: List[Document] = []
            try:
                with open(output_buffer, "r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            documents.append(dict_to_doc(json.loads(line)))
                logger.info("✅ Streaming complete: %d total chunks extracted", len(documents))
            except FileNotFoundError:
                logger.warning("No buffer file found; streaming produced 0 chunks")
            finally:
                # Clean up temp buffer after loading
                output_buffer.unlink(missing_ok=True)
            return documents

    def _process_marker_batch(
        self,
        batch_pdf: Path,
        doc_id: str,
        chunk_start_idx: int,
        language: str,
        method: str,
        retry: int = 1,
    ) -> List[Document]:
        """Process a single batch PDF with Marker (detect + recognize)."""
        for attempt in range(1, retry + 2):
            try:
                with cuda_memory_guard(f"marker_batch:{batch_pdf.name}"):
                    # CRITICAL: Use inference_mode to disable gradients during inference
                    with torch.inference_mode():
                        rendered = self._models.marker_converter(str(batch_pdf))
                        result = text_from_rendered(rendered)
                        text = result[0] if isinstance(result, (list, tuple)) else result
                    
                    # MASSIVE FIX: Explicitly destroy tensors to free VRAM immediately
                    del rendered
                    del result
                    gc.collect()
            except CudaOOMError:
                logger.error(
                    "CUDA OOM on batch attempt %d/%d. Tearing down Marker and retrying…",
                    attempt, retry + 1,
                )
                self._models.teardown_marker()
                hard_cuda_reset()  # Use hard reset instead of soft
                if attempt > retry:
                    logger.error("All retries exhausted for batch — returning empty")
                    return []
                time.sleep(2)
                continue
            except Exception as exc:
                logger.error("Marker batch processing failed: %s", exc)
                return []
            break

        if not text or not text.strip():
            logger.warning("Marker batch returned empty text")
            return []

        # Split into chunks using markdown hierarchy
        md_docs = self._models.md_splitter.split_text(text)
        
        # CLEANUP: Delete text to free memory
        del text
        gc.collect()
        
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
                    "chunk_id": generate_chunk_id(doc_id, chunk_start_idx + idx),
                    "source": str(batch_pdf.parent / batch_pdf.stem.split("_batch_")[0]),
                    "filename": str(batch_pdf.parent / batch_pdf.stem.split("_batch_")[0]),
                    "subject": batch_pdf.parent.name,
                    "language": language,
                    "extraction_method": method,
                    "heading": heading,
                    "page": f"vision_batch_{idx}",
                },
            ))

        return documents

    def _slow_path_single(
        self,
        pdf_path: Path,
        language: str,
        method: str,
        retry: int = 1,
    ) -> List[Document]:
        """Fallback: Process entire PDF at once (old v2 behavior)."""
        logger.info("🏎️  GPU path (Marker fallback, full PDF): %s [lang=%s]", pdf_path.name, language)

        for attempt in range(1, retry + 2):
            try:
                with cuda_memory_guard(f"marker:{pdf_path.name}"):
                    with torch.inference_mode():
                        rendered = self._models.marker_converter(str(pdf_path))
                        result = text_from_rendered(rendered)
                        text = result[0] if isinstance(result, (list, tuple)) else result
                    
                    # MASSIVE FIX: Explicitly destroy tensors
                    del rendered
                    del result
                    gc.collect()
            except CudaOOMError:
                logger.error(
                    "CUDA OOM on attempt %d/%d for %s. Tearing down Marker and retrying…",
                    attempt, retry + 1, pdf_path.name,
                )
                self._models.teardown_marker()
                hard_cuda_reset()  # Use hard reset
                if attempt > retry:
                    logger.error("All retries exhausted for %s — skipping.", pdf_path.name)
                    return []
                time.sleep(2)
                continue
            except Exception as exc:
                logger.error("Marker failed on %s: %s", pdf_path.name, exc)
                return []
            break

        if not text or not text.strip():
            logger.warning("Marker returned empty text for %s", pdf_path.name)
            return []

        md_docs = self._models.md_splitter.split_text(text)
        
        # CLEANUP: Delete text immediately
        del text
        gc.collect()
        
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

    def extract_file(self, pdf_path_str: str, language: str = "english", output_file: Optional[str] = None) -> List[Document]:
        """Extract a single PDF. If output_file provided, streams chunks directly there (for streaming paths)."""
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path_str}")

        lang = language.strip().lower()
        try:
            if lang in ("en", "english"):
                return self._fast_path(pdf_path)
            elif lang in ("te", "telugu"):
                # Streaming path: returns int (chunk count), chunks already on disk
                chunk_count = self._slow_path_streaming(pdf_path, language="telugu", method="marker_vision_streaming", output_file=output_file)
                logger.info("Telugu streaming produced %d chunks (on disk at %s)", chunk_count, output_file or "temp buffer")
                return []  # Empty list; chunks already flushed to disk
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
        """Walk *data_dir* recursively, extract all PDFs, and write one JSON file per PDF."""
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
            
            # For streaming paths (telugu), pass output_file so chunks go directly there
            # For fast paths (english), extract_file returns docs to write
            if detected_lang.lower() in ("te", "telugu"):
                # Streaming path: chunks written directly to file
                chunk_count = self.extract_file(
                    str(pdf_path), 
                    language=detected_lang, 
                    output_file=str(save_file)
                )
                # For streaming, extract_file returns [], so count chunks from file
                try:
                    with open(save_file, "r", encoding="utf-8") as fh:
                        chunk_count = sum(1 for line in fh if line.strip())
                    docs = []
                except FileNotFoundError:
                    chunk_count = 0
                    docs = []
            else:
                # Fast path: extract_file returns docs to write
                docs = self.extract_file(str(pdf_path), language=detected_lang)
                chunk_count = len(docs)
            
            elapsed = time.perf_counter() - t0

            if chunk_count == 0:
                logger.warning("❌ No chunks produced for %s", pdf_path.name)
                stats["failed"] += 1
                continue

            # For non-streaming paths, write docs to file
            if detected_lang.lower() not in ("te", "telugu"):
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
            stats["total_chunks"] += chunk_count
            logger.info(
                "✅ %s → %d chunks [%.1fs]", pdf_path.name, chunk_count, elapsed
            )

            # BIGGEST FIX: Tear down Marker models after EVERY PDF (not every 20)
            # This prevents VRAM fragmentation and model drift
            logger.info("🧹 Full VRAM reset after each PDF…")
            self._models.teardown_marker()
            hard_cuda_reset()
            time.sleep(1)  # Let OS settle
            log_vram_status("Post-PDF cleanup")

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
    # Log initial system state
    logger.info("=" * 70)
    logger.info("PDF Extraction Pipeline v3.1 (Memory-Safe with Auto-Throttling)")
    logger.info("=" * 70)
    log_vram_status("Initial VRAM")
    
    # Initialize with safe default batch size
    extractor = ProductionPDFExtractor(marker_batch_pages=MARKER_BATCH_PAGES)
    
    summary = extractor.process_directory(
        data_dir="data",
        output_dir="extracted_chunks",
        force=False,
        vram_relief_every=20,
    )
    
    log_vram_status("Final VRAM")
    logger.info("=" * 70)
    print(json.dumps(summary, indent=2))


"""
Production PDF Extraction Pipeline v4 — Stable VRAM Edition
===========================================================

MAJOR FIXES:
- True CUDA cleanup after EVERY PDF
- inference_mode() everywhere
- aggressive tensor destruction
- reduced CUDA fragmentation
- Docling forced to CPU
- proper GC timing
- stable long-running processing
- optimized for RTX 4060 8GB

Recommended:
RTX 4060 8GB
32GB RAM
Python 3.13
"""

# import gc
# import os
# import re
# import json
# import time
# import logging
# import unicodedata
# import hashlib
# import traceback
# import tempfile

# from contextlib import contextmanager
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Tuple

# # =============================================================================
# # ENVIRONMENT CONFIG
# # =============================================================================

# # CUDA SETTINGS
# os.environ["TORCH_DEVICE"] = "cuda"
# os.environ["TORCH_DTYPE"] = "fp16"

# # HUGE FIX: lower split size for 8GB GPU
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
#     "expandable_segments:True,max_split_size_mb:64"
# )

# # Force Docling to CPU
# os.environ["DOCLING_DEVICE"] = "cpu"

# # OCR batch sizes
# os.environ["SURYA_DET_BATCH_SIZE"] = "1"
# os.environ["SURYA_REC_BATCH_SIZE"] = "1"
# os.environ["TEXIFY_BATCH_SIZE"] = "1"

# # Thread optimization
# _cpu_cores = os.cpu_count() or 1
# _thread_count = str(min(12, max(1, _cpu_cores - 2)))

# os.environ["OMP_NUM_THREADS"] = _thread_count
# os.environ["MKL_NUM_THREADS"] = _thread_count

# # =============================================================================
# # IMPORTS
# # =============================================================================

# import torch

# # MASSIVE VRAM SAVER
# torch.set_grad_enabled(False)

# from langchain_core.documents import Document
# from langchain_text_splitters import MarkdownHeaderTextSplitter

# from docling.document_converter import DocumentConverter
# from docling.chunking import HierarchicalChunker

# from marker.converters.pdf import PdfConverter
# from marker.models import create_model_dict
# from marker.output import text_from_rendered

# # =============================================================================
# # LOGGING
# # =============================================================================

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# logger = logging.getLogger("PDFExtractor")

# # =============================================================================
# # CONSTANTS
# # =============================================================================

# MIN_CHUNK_CHARS = 40

# MARKER_BATCH_PAGES = 4

# HEADERS_TO_SPLIT = [
#     ("#", "Header 1"),
#     ("##", "Header 2"),
#     ("###", "Header 3"),
# ]

# LIGATURE_MAP = str.maketrans({
#     "\ufb00": "ff",
#     "\ufb01": "fi",
#     "\ufb02": "fl",
#     "\ufb03": "ffi",
#     "\ufb04": "ffl",
#     "\u2019": "'",
#     "\u2018": "'",
#     "\u201c": '"',
#     "\u201d": '"',
#     "\u2013": "-",
#     "\u2014": "-",
#     "\u00ad": "",
#     "\u00a0": " ",
# })

# # =============================================================================
# # CUDA HELPERS
# # =============================================================================

# def cuda_available() -> bool:
#     return torch.cuda.is_available()


# def hard_cuda_reset():
#     """
#     REAL CUDA cleanup.
#     Much stronger than empty_cache().
#     """

#     gc.collect()

#     if torch.cuda.is_available():

#         torch.cuda.empty_cache()

#         try:
#             torch.cuda.ipc_collect()
#         except Exception:
#             pass

#         try:
#             torch.cuda.reset_peak_memory_stats()
#         except Exception:
#             pass

#         torch.cuda.synchronize()


# def free_cuda_memory():
#     gc.collect()

#     if cuda_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()


# def get_vram_stats() -> Tuple[float, float, float]:

#     if not cuda_available():
#         return (0.0, 0.0, 100.0)

#     free_bytes, total_bytes = torch.cuda.mem_get_info()

#     used_bytes = total_bytes - free_bytes

#     used_gb = used_bytes / (1024 ** 3)
#     total_gb = total_bytes / (1024 ** 3)

#     free_percent = (free_bytes / total_bytes) * 100

#     return used_gb, total_gb, free_percent


# def log_vram(label=""):

#     used, total, free = get_vram_stats()

#     logger.info(
#         "%s VRAM %.2f / %.2f GB (%.1f%% free)",
#         label,
#         used,
#         total,
#         free
#     )


# class CudaOOMError(RuntimeError):
#     pass


# @contextmanager
# def cuda_memory_guard(label=""):

#     try:
#         yield

#     except torch.cuda.OutOfMemoryError as exc:

#         hard_cuda_reset()

#         raise CudaOOMError(
#             f"CUDA OOM during {label}"
#         ) from exc

#     finally:
#         free_cuda_memory()

# # =============================================================================
# # UTILITIES
# # =============================================================================

# def clean_pdf_text(text: str) -> str:

#     if not isinstance(text, str):
#         return ""

#     text = re.sub(r"[\uE000-\uF8FF]", "", text)

#     text = text.translate(LIGATURE_MAP)

#     text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


# def generate_doc_id(pdf_path: Path) -> str:

#     key = str(pdf_path).lower()

#     return hashlib.md5(
#         key.encode()
#     ).hexdigest()[:12]


# def generate_chunk_id(doc_id: str, idx: int):

#     return f"{doc_id}-{idx:04d}"


# def doc_to_dict(doc: Document):

#     return {
#         "page_content": doc.page_content,
#         "metadata": doc.metadata
#     }

# # =============================================================================
# # PDF HELPERS
# # =============================================================================

# def get_pdf_page_count(pdf_path: str):

#     import pypdf

#     with open(pdf_path, "rb") as fh:

#         reader = pypdf.PdfReader(fh)

#         return len(reader.pages)


# def extract_pdf_page_range(
#     pdf_path,
#     output_pdf,
#     page_start,
#     page_end
# ):

#     import pypdf

#     with open(pdf_path, "rb") as fh:

#         reader = pypdf.PdfReader(fh)

#         writer = pypdf.PdfWriter()

#         for i in range(page_start, page_end):
#             writer.add_page(reader.pages[i])

#         with open(output_pdf, "wb") as out_fh:
#             writer.write(out_fh)

# # =============================================================================
# # MODEL MANAGER
# # =============================================================================

# class ModelManager:

#     def __init__(self):

#         self._docling_converter = None
#         self._docling_chunker = None

#         self._marker_converter = None
#         self._marker_artifacts = None

#         self._md_splitter = MarkdownHeaderTextSplitter(
#             HEADERS_TO_SPLIT
#         )

#     @property
#     def docling_converter(self):

#         if self._docling_converter is None:

#             logger.info("Loading Docling...")

#             self._docling_converter = DocumentConverter()

#         return self._docling_converter

#     @property
#     def docling_chunker(self):

#         if self._docling_chunker is None:

#             self._docling_chunker = HierarchicalChunker()

#         return self._docling_chunker

#     @property
#     def marker_converter(self):

#         if self._marker_converter is None:

#             logger.info("Loading Marker models...")

#             with cuda_memory_guard("marker_load"):

#                 self._marker_artifacts = create_model_dict()

#                 self._marker_converter = PdfConverter(
#                     artifact_dict=self._marker_artifacts
#                 )

#         return self._marker_converter

#     @property
#     def md_splitter(self):
#         return self._md_splitter

#     def teardown_marker(self):

#         logger.info("Destroying Marker models...")

#         self._marker_converter = None
#         self._marker_artifacts = None

#         gc.collect()

#         hard_cuda_reset()

# # =============================================================================
# # MAIN EXTRACTOR
# # =============================================================================

# class ProductionPDFExtractor:

#     def __init__(self):

#         self.models = ModelManager()

#     # =========================================================================
#     # FAST PATH
#     # =========================================================================

#     def fast_extract(self, pdf_path: Path):

#         logger.info("Fast path: %s", pdf_path.name)

#         result = self.models.docling_converter.convert(
#             str(pdf_path)
#         )

#         chunks = list(
#             self.models.docling_chunker.chunk(
#                 result.document
#             )
#         )

#         documents = []

#         doc_id = generate_doc_id(pdf_path)

#         for idx, chunk in enumerate(chunks):

#             clean = clean_pdf_text(chunk.text)

#             if len(clean) < MIN_CHUNK_CHARS:
#                 continue

#             documents.append(
#                 Document(
#                     page_content=clean,
#                     metadata={
#                         "doc_id": doc_id,
#                         "chunk_id": generate_chunk_id(doc_id, idx),
#                         "source": str(pdf_path),
#                         "filename": pdf_path.name,
#                         "method": "docling"
#                     }
#                 )
#             )

#         return documents

#     # =========================================================================
#     # MARKER BATCH PROCESSING
#     # =========================================================================

#     def process_marker_batch(
#         self,
#         batch_pdf: Path,
#         doc_id: str,
#         start_idx: int
#     ):

#         try:

#             with cuda_memory_guard("marker_batch"):

#                 with torch.inference_mode():

#                     rendered = self.models.marker_converter(
#                         str(batch_pdf)
#                     )

#                     result = text_from_rendered(rendered)

#                     text = (
#                         result[0]
#                         if isinstance(result, (list, tuple))
#                         else result
#                     )

#                 # MASSIVE FIX
#                 del rendered
#                 del result

#                 gc.collect()

#         except Exception as exc:

#             logger.error("Marker failed: %s", exc)

#             return []

#         if not text:
#             return []

#         md_docs = self.models.md_splitter.split_text(text)

#         del text

#         gc.collect()

#         documents = []

#         for idx, md_doc in enumerate(md_docs):

#             clean = clean_pdf_text(
#                 md_doc.page_content
#             )

#             if len(clean) < MIN_CHUNK_CHARS:
#                 continue

#             documents.append(
#                 Document(
#                     page_content=clean,
#                     metadata={
#                         "doc_id": doc_id,
#                         "chunk_id": generate_chunk_id(
#                             doc_id,
#                             start_idx + idx
#                         ),
#                         "source": str(batch_pdf),
#                         "method": "marker"
#                     }
#                 )
#             )

#         return documents

#     # =========================================================================
#     # STREAMING PROCESSOR
#     # =========================================================================

#     def streaming_extract(
#         self,
#         pdf_path: Path,
#         output_file: str
#     ):

#         total_pages = get_pdf_page_count(
#             str(pdf_path)
#         )

#         logger.info(
#             "Streaming %d pages...",
#             total_pages
#         )

#         doc_id = generate_doc_id(pdf_path)

#         chunk_counter = 0

#         output_path = Path(output_file)

#         output_path.parent.mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         temp_dir = tempfile.gettempdir()

#         for page_start in range(
#             0,
#             total_pages,
#             MARKER_BATCH_PAGES
#         ):

#             page_end = min(
#                 page_start + MARKER_BATCH_PAGES,
#                 total_pages
#             )

#             log_vram(f"Before batch {page_start}")

#             temp_pdf = (
#                 Path(temp_dir)
#                 / f"batch_{page_start}.pdf"
#             )

#             extract_pdf_page_range(
#                 str(pdf_path),
#                 str(temp_pdf),
#                 page_start,
#                 page_end
#             )

#             batch_docs = self.process_marker_batch(
#                 temp_pdf,
#                 doc_id,
#                 chunk_counter
#             )

#             chunk_counter += len(batch_docs)

#             with open(
#                 output_path,
#                 "a",
#                 encoding="utf-8"
#             ) as fh:

#                 for doc in batch_docs:

#                     fh.write(
#                         json.dumps(
#                             doc_to_dict(doc),
#                             ensure_ascii=False
#                         ) + "\n"
#                     )

#             # CRITICAL CLEANUP
#             del batch_docs

#             gc.collect()

#             temp_pdf.unlink(missing_ok=True)

#             hard_cuda_reset()

#             log_vram(f"After batch {page_start}")

#         return chunk_counter

#     # =========================================================================
#     # SINGLE FILE
#     # =========================================================================

#     def process_pdf(
#         self,
#         pdf_path: Path,
#         output_file: str,
#         language="english"
#     ):

#         logger.info("=" * 70)
#         logger.info("Processing %s", pdf_path.name)

#         try:

#             if language == "english":

#                 docs = self.fast_extract(pdf_path)

#                 with open(
#                     output_file,
#                     "w",
#                     encoding="utf-8"
#                 ) as fh:

#                     json.dump(
#                         [doc_to_dict(d) for d in docs],
#                         fh,
#                         ensure_ascii=False,
#                         indent=2
#                     )

#                 count = len(docs)

#                 del docs

#             else:

#                 count = self.streaming_extract(
#                     pdf_path,
#                     output_file
#                 )

#             logger.info(
#                 "Finished %s -> %d chunks",
#                 pdf_path.name,
#                 count
#             )

#             return count

#         finally:

#             # BIGGEST FIX
#             logger.info("FULL VRAM RESET")

#             self.models.teardown_marker()

#             hard_cuda_reset()

#             time.sleep(2)

#             log_vram("Post-cleanup")

#     # =========================================================================
#     # DIRECTORY PROCESSING
#     # =========================================================================

#     def process_directory(
#         self,
#         data_dir,
#         output_dir="extracted_chunks"
#     ):

#         base = Path(data_dir)

#         out = Path(output_dir)

#         out.mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         pdfs = sorted(base.rglob("*.pdf"))

#         logger.info(
#             "Found %d PDFs",
#             len(pdfs)
#         )

#         stats = {
#             "processed": 0,
#             "failed": 0,
#             "chunks": 0
#         }

#         for pdf in pdfs:

#             try:

#                 lang = (
#                     "telugu"
#                     if "telugu" in str(pdf).lower()
#                     else "english"
#                 )

#                 doc_id = generate_doc_id(pdf)

#                 save_file = (
#                     out /
#                     f"{doc_id}_{pdf.stem}.json"
#                 )

#                 chunks = self.process_pdf(
#                     pdf,
#                     str(save_file),
#                     lang
#                 )

#                 stats["processed"] += 1
#                 stats["chunks"] += chunks

#             except Exception as exc:

#                 logger.error(
#                     "FAILED %s: %s",
#                     pdf.name,
#                     exc
#                 )

#                 traceback.print_exc()

#                 stats["failed"] += 1

#                 hard_cuda_reset()

#         return stats

# # =============================================================================
# # ENTRYPOINT
# # =============================================================================

# if __name__ == "__main__":

#     logger.info("=" * 70)
#     logger.info("PDF Extraction Pipeline v4")
#     logger.info("=" * 70)

#     log_vram("Initial")

#     extractor = ProductionPDFExtractor()

#     summary = extractor.process_directory(
#         data_dir="data",
#         output_dir="extracted_chunks"
#     )

#     logger.info("=" * 70)

#     log_vram("Final")

#     print(json.dumps(summary, indent=2))
