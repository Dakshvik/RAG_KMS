"""
Production PDF Extraction Pipeline
===================================
Optimized for RTX 4060 8GB | Windows | Multi-language (English + Telugu)

Key improvements over v1:
  - Zero VRAM spikes: models loaded lazily and released after each file
  - Retrieval-quality chunks: semantic paragraph merging, overlap, rich metadata
  - Future-proof: strategy pattern, dataclasses, typed interfaces throughout
  - Memory-safe: context managers, explicit del + gc, no persistent GPU tensors
  - Resilient: per-file error isolation, checkpointing, structured logging

v2 fixes:
  - Removed expandable_segments (unsupported on Windows CUDA allocator)
  - Fixed scatter() dtype mismatch: models explicitly cast to fp16 after load
  - Sequential model offloading to stay within 8 GB VRAM budget
  - SURYA_BATCH_SIZE reduced further for large PDFs
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Generator, List, Optional, Protocol

# ── Environment flags (must be set before ANY torch / surya / marker import) ─

# Windows CUDA allocator: expandable_segments is Linux-only, never use on Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# ── Surya batch sizes (OCR backbone inside Marker) ───────────────────────────
# Must be set BEFORE marker/surya is imported anywhere.
os.environ["SURYA_BATCH_SIZE"]       = "1"   # layout pages per batch — safest for 8 GB
os.environ["RECOGNITION_BATCH_SIZE"] = "16"  # text recognition batch
os.environ["DETECTOR_BATCH_SIZE"]    = "2"   # text detection batch
os.environ["ORDER_BATCH_SIZE"]       = "4"   # reading-order model batch

# Saturate 20-core CPU (leave 4 for OS / GPU handoffs)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "16")

import torch  # noqa: E402  (must come after env vars)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pdf-extractor")


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA MODELS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkMetadata:
    doc_id: str
    chunk_id: str
    source: str
    filename: str
    subject: str
    language: str
    extraction_method: str
    heading: str
    page: int | str
    chunk_index: int
    total_chunks: int          # set in post-processing pass
    prev_chunk_id: str = ""    # for retrieval continuity
    next_chunk_id: str = ""    # for retrieval continuity
    char_count: int = 0


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        return {"page_content": self.text, "metadata": asdict(self.metadata)}


# ═════════════════════════════════════════════════════════════════════════════
# 2. ID HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def make_doc_id(pdf_path: Path) -> str:
    subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
    key = f"{subject}::{pdf_path.name}".lower()
    return hashlib.md5(key.encode()).hexdigest()[:12]


def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}-{index:04d}"


# ═════════════════════════════════════════════════════════════════════════════
# 3. TEXT CLEANING  (retrieval-aware)
# ═════════════════════════════════════════════════════════════════════════════

_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00ad": "", "\u00a0": " ",
})

# Boilerplate patterns that pollute retrieval recall
_BOILERPLATE_RE = re.compile(
    r"(?im)"
    r"(^\s*page\s+\d+\s*(of\s+\d+)?\s*$"       # "Page 3 of 10"
    r"|^\s*\d{1,4}\s*$"                           # bare page numbers
    r"|^\s*(table\s+of\s+contents|references)\s*$"  # section stubs
    r")"
)


def clean_text(text: str, language: str = "english") -> str:
    if not isinstance(text, str):
        return ""

    # Remove private-use Unicode (broken PDF fonts)
    text = re.sub(r"[\uE000-\uF8FF]", "", text)

    # Expand ligatures and fix smart quotes / dashes
    text = text.translate(_LIGATURES)

    # Re-join hyphenated line breaks (common in PDFs)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Strip boilerplate lines
    text = _BOILERPLATE_RE.sub("", text)

    # Normalise whitespace (preserve paragraph breaks as double newline)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ASCII-only normalisation for strictly English content
    if language == "english":
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", errors="ignore").decode("ascii")

    return text.strip()


# ─── Corruption detector ────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "the be to of and a in that have it is for on with he as you do at".split()
)


def text_is_corrupted(text: str, threshold: float = 0.08) -> bool:
    """Return True when English stop-word density is below threshold."""
    words = re.findall(r"\b[a-z]{2,}\b", text.lower())
    if len(words) < 20:
        return False
    ratio = sum(1 for w in words if w in _STOP_WORDS) / len(words)
    return ratio < threshold


# ═════════════════════════════════════════════════════════════════════════════
# 4. CHUNKING STRATEGY  (retrieval-quality)
# ═════════════════════════════════════════════════════════════════════════════

# Tuneable constants — tweak without touching logic
MIN_CHUNK_CHARS = 200     # discard micro-fragments
TARGET_CHUNK_CHARS = 1200 # sweet spot for dense-passage retrieval
MAX_CHUNK_CHARS = 2000    # hard ceiling before forced split
OVERLAP_CHARS = 150       # sentence-level overlap for context continuity


def _split_into_sentences(text: str) -> List[str]:
    """Naive but fast sentence splitter (no NLTK dependency)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _merge_to_target(
    paragraphs: List[str],
    target: int = TARGET_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> List[str]:
    """
    Merge short paragraphs into retrieval-sized windows with sentence overlap.

    Strategy
    --------
    Accumulate paragraphs until we hit *target*.  When we flush, carry the
    last *overlap* characters into the next window so retrieval never lands
    between a question and its answer.
    """
    chunks: List[str] = []
    buffer = ""
    tail = ""  # overlap tail from previous chunk

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        candidate = (tail + " " + buffer + " " + para).strip() if buffer else (tail + " " + para).strip()

        if len(candidate) >= max_chars:
            # Force-flush current buffer before adding the oversized paragraph
            if buffer:
                chunks.append((tail + " " + buffer).strip())
                tail = buffer[-overlap:] if len(buffer) > overlap else buffer
                buffer = ""
            # Split the big paragraph at sentence boundaries
            sentences = _split_into_sentences(para)
            sub_buf = tail
            for sent in sentences:
                if len(sub_buf) + len(sent) + 1 > max_chars:
                    if sub_buf.strip():
                        chunks.append(sub_buf.strip())
                    tail = sub_buf[-overlap:] if len(sub_buf) > overlap else sub_buf
                    sub_buf = tail + " " + sent
                else:
                    sub_buf = (sub_buf + " " + sent).strip()
            buffer = sub_buf
        elif len(candidate) >= target:
            chunks.append(candidate.strip())
            # Overlap: keep last N chars (prefer sentence boundary)
            sents = _split_into_sentences(candidate)
            tail = sents[-1] if sents and len(sents[-1]) <= overlap else candidate[-overlap:]
            buffer = ""
        else:
            buffer = candidate

    # Flush remainder
    leftover = (tail + " " + buffer).strip() if buffer else ""
    if leftover:
        chunks.append(leftover)

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


# ═════════════════════════════════════════════════════════════════════════════
# 5. VRAM CONTEXT MANAGER
# ═════════════════════════════════════════════════════════════════════════════

@contextmanager
def vram_guard(label: str = "") -> Generator[None, None, None]:
    """
    Flush GPU allocator before and after a block.
    Logs VRAM usage at INFO level so you can see peak consumption per file.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            reserved_mb = torch.cuda.memory_reserved() / 1024 ** 2
            log.info(
                f"[{label}] VRAM peak={peak_mb:.0f} MB  reserved-after-flush={reserved_mb:.0f} MB"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 6. EXTRACTION STRATEGIES  (Strategy Pattern — easy to extend)
# ═════════════════════════════════════════════════════════════════════════════

@contextmanager
def _no_cuda() -> Generator[None, None, None]:
    """
    Temporarily hide CUDA from PyTorch for the duration of the block.

    Why this is necessary
    ---------------------
    Docling's DocLayNet (layout) and TableFormer (table) models call
    torch.cuda.is_available() internally and silently self-migrate to GPU
    even when AcceleratorOptions(device='cpu') is passed. On a 232-page PDF
    this burns 4-6 GB VRAM before Marker ever starts.

    Setting CUDA_VISIBLE_DEVICES='' makes torch.cuda.is_available() return
    False for the whole process, preventing any Docling GPU allocation.
    We restore the original value immediately after so Marker can still use
    CUDA normally.
    """
    original = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


class ExtractionStrategy(Protocol):
    def extract(self, pdf_path: Path, language: str) -> List[Chunk]: ...


# ── 6a. Fast path: Docling — hard-locked to CPU ─────────────────────────────

class DoclingStrategy:
    """
    CPU-only extraction via Docling + HierarchicalChunker.

    Docling's layout/table neural models (DocLayNet, TableFormer) silently
    migrate to CUDA even when device='cpu' is set in AcceleratorOptions —
    they check is_available() at forward-pass time, not at init.

    Fix: we initialise the converter inside _no_cuda() which sets
    CUDA_VISIBLE_DEVICES='' so is_available() returns False throughout
    the entire Docling init and conversion call.  CUDA_VISIBLE_DEVICES is
    restored before Marker runs so the GPU is fully available to it.
    """

    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            AcceleratorOptions,
        )
        from docling.chunking import HierarchicalChunker

        log.info("Loading Docling (CPU-only, CUDA hidden)…")

        with _no_cuda():
            opts = PdfPipelineOptions()
            opts.do_ocr = False
            # Disable the two GPU-hungry sub-models entirely for text PDFs
            opts.do_table_structure = False   # TableFormer — not needed for plain text
            opts.accelerator_options = AcceleratorOptions(num_threads=16, device="cpu")

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
            )
        self._chunker = HierarchicalChunker()

    def extract(self, pdf_path: Path, language: str) -> List[Chunk]:
        doc_id = make_doc_id(pdf_path)
        subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem

        # Run conversion with CUDA hidden so no model can sneak onto GPU
        with _no_cuda():
            doc = self._converter.convert(str(pdf_path)).document
        raw_chunks = list(self._chunker.chunk(doc))

        # Corruption check on first few chunks
        sample = " ".join(c.text for c in raw_chunks[:5])
        if text_is_corrupted(sample):
            raise ValueError("Docling output appears corrupted — caller should reroute to Marker")

        # Collect (heading, page, raw_text) triples
        triples: List[tuple[str, int | str, str]] = []
        for chunk in raw_chunks:
            text = clean_text(chunk.text, language)
            if not text:
                continue
            meta = chunk.meta
            heading = meta.headings[0] if meta and meta.headings else "Unknown"
            page = 1
            if meta and meta.doc_items:
                prov = meta.doc_items[0].prov
                if prov:
                    page = prov[0].page_no
            triples.append((heading, page, text))

        return _triples_to_chunks(triples, doc_id, subject, str(pdf_path), language, "docling_fast")


# ── 6b. Slow path: Marker on CUDA ───────────────────────────────────────────

class MarkerStrategy:
    """
    GPU-accelerated extraction via Marker (vision + OCR).

    VRAM budget strategy for RTX 4060 8 GB (Windows)
    --------------------------------------------------
    - Models are loaded fresh per file and fully deleted after.
    - Every sub-model is explicitly cast to fp16 *after* loading to avoid the
      scatter() dtype mismatch that occurs when Marker mixes fp32 init weights
      with fp16 autocast during forward passes.
    - torch.autocast is NOT used here — it is redundant (and buggy) when the
      model weights are already fp16. Instead we cast once and run in fp16.
    - SURYA batch sizes are controlled via env vars set at module top.
    """

    @staticmethod
    def _cast_to_fp16(artifact_dict: dict) -> None:
        """
        Cast every nn.Module inside artifact_dict to fp16 in-place.
        This is the fix for: scatter(): Expected self.dtype == src.dtype
        (which happens when model params are fp32 but inputs are fp16).
        """
        for name, obj in artifact_dict.items():
            if isinstance(obj, torch.nn.Module):
                try:
                    obj.half()   # fp16 in-place
                    log.debug(f"  Cast {name} → fp16")
                except Exception as e:
                    log.warning(f"  Could not cast {name} to fp16: {e}")

    def extract(self, pdf_path: Path, language: str) -> List[Chunk]:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        doc_id = make_doc_id(pdf_path)
        subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
        text = ""

        with vram_guard("marker"):
            artifact_dict = None
            converter = None
            rendered = None
            try:
                log.info(f"  Loading Marker models → VRAM  [{pdf_path.name}]")
                artifact_dict = create_model_dict()

                # ── KEY FIX: cast all sub-models to fp16 before any forward pass ──
                self._cast_to_fp16(artifact_dict)

                converter = PdfConverter(artifact_dict=artifact_dict)

                # Run inference — no autocast needed; models are already fp16
                rendered = converter(str(pdf_path))
                text, _, _ = text_from_rendered(rendered)

            except Exception as exc:
                log.error(f"Marker failed on {pdf_path.name}: {exc}", exc_info=True)
                return []
            finally:
                # Delete in reverse construction order; vram_guard handles cache flush
                if rendered is not None:
                    del rendered
                if converter is not None:
                    del converter
                if artifact_dict is not None:
                    del artifact_dict

        if not text.strip():
            return []

        return _markdown_to_chunks(text, doc_id, subject, str(pdf_path), language, "marker_vision")


# ═════════════════════════════════════════════════════════════════════════════
# 7. SHARED CHUNK ASSEMBLY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _triples_to_chunks(
    triples: List[tuple[str, int | str, str]],
    doc_id: str,
    subject: str,
    source: str,
    language: str,
    method: str,
) -> List[Chunk]:
    """Convert (heading, page, text) triples → retrieval-quality Chunk list."""

    # Group contiguous same-heading paragraphs, then merge to target size
    groups: List[tuple[str, int | str, List[str]]] = []
    for heading, page, text in triples:
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
        if groups and groups[-1][0] == heading:
            groups[-1][2].extend(paragraphs)
        else:
            groups.append((heading, page, paragraphs))

    raw_chunks: List[tuple[str, int | str, str]] = []
    for heading, page, paragraphs in groups:
        for merged in _merge_to_target(paragraphs):
            raw_chunks.append((heading, page, merged))

    return _finalise(raw_chunks, doc_id, subject, source, language, method)


def _markdown_to_chunks(
    md_text: str,
    doc_id: str,
    subject: str,
    source: str,
    language: str,
    method: str,
) -> List[Chunk]:
    """Parse Marker markdown output → retrieval-quality Chunk list."""

    # Split on markdown headers to get (heading, body) pairs
    header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    positions = [(m.start(), m.group(2).strip()) for m in header_re.finditer(md_text)]
    positions.append((len(md_text), "__END__"))

    raw_chunks: List[tuple[str, int | str, str]] = []
    for i, (start, heading) in enumerate(positions[:-1]):
        body = md_text[start: positions[i + 1][0]]
        # Strip the header line itself
        body = header_re.sub("", body, count=1).strip()
        body = clean_text(body, language)
        paragraphs = [p.strip() for p in re.split(r"\n\n+", body) if p.strip()]
        for merged in _merge_to_target(paragraphs):
            raw_chunks.append((heading, "vision", merged))

    # Handle content before the first header
    if positions[0][0] > 0:
        preamble = md_text[: positions[0][0]]
        preamble = clean_text(preamble, language)
        paragraphs = [p.strip() for p in re.split(r"\n\n+", preamble) if p.strip()]
        for merged in _merge_to_target(paragraphs):
            raw_chunks.insert(0, ("Preamble", "vision", merged))

    return _finalise(raw_chunks, doc_id, subject, source, language, method)


def _finalise(
    raw: List[tuple[str, int | str, str]],
    doc_id: str,
    subject: str,
    source: str,
    language: str,
    method: str,
) -> List[Chunk]:
    """Attach IDs, prev/next pointers, and char counts; return Chunk list."""
    filename = Path(source).name
    chunks: List[Chunk] = []

    for idx, (heading, page, text) in enumerate(raw):
        chunk_id = make_chunk_id(doc_id, idx)
        meta = ChunkMetadata(
            doc_id=doc_id,
            chunk_id=chunk_id,
            source=source.replace("\\", "/"),
            filename=filename,
            subject=subject,
            language=language,
            extraction_method=method,
            heading=heading,
            page=page,
            chunk_index=idx,
            total_chunks=len(raw),  # final value set below
            char_count=len(text),
        )
        chunks.append(Chunk(text=text, metadata=meta))

    # Second pass: wire prev/next and set total_chunks
    for i, chunk in enumerate(chunks):
        chunk.metadata.total_chunks = len(chunks)
        chunk.metadata.prev_chunk_id = chunks[i - 1].metadata.chunk_id if i > 0 else ""
        chunk.metadata.next_chunk_id = chunks[i + 1].metadata.chunk_id if i < len(chunks) - 1 else ""

    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# 8. LANGUAGE DETECTOR  (lightweight, no external model needed)
# ═════════════════════════════════════════════════════════════════════════════

# Telugu Unicode block: U+0C00–U+0C7F
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")


def detect_language(pdf_path: Path) -> str:
    """
    Heuristic: check directory name first (fast), then peek at file name.
    Extend this with a proper langdetect call if needed.
    """
    parts_lower = [p.lower() for p in pdf_path.parts]
    if "telugu" in parts_lower:
        return "telugu"
    if "english" in parts_lower:
        return "english"
    # Could do a quick text peek here with pdfminer if needed
    return "english"


# ═════════════════════════════════════════════════════════════════════════════
# 9. ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

class PDFExtractor:
    """
    Single public entry point.

    Design decisions
    ----------------
    - DoclingStrategy is instantiated once (stateless, CPU-only).
    - MarkerStrategy is stateless; it loads models per-file and destroys them.
    - VRAM is always at baseline between files.
    """

    def __init__(self) -> None:
        self._docling = DoclingStrategy()
        self._marker = MarkerStrategy()

    # ── Public API ───────────────────────────────────────────────────────────

    def extract_file(self, pdf_path: Path, language: Optional[str] = None) -> List[Chunk]:
        lang = language or detect_language(pdf_path)

        if lang == "english":
            try:
                return self._docling.extract(pdf_path, lang)
            except ValueError:
                log.warning(f"Docling corruption detected in {pdf_path.name} — falling back to Marker")
                return self._marker.extract(pdf_path, lang)
        else:
            return self._marker.extract(pdf_path, lang)

    def process_directory(
        self,
        data_dir: str | Path,
        output_dir: str | Path = "extracted_chunks",
    ) -> None:
        base = Path(data_dir)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(base.rglob("*.pdf"))
        log.info(f"Found {len(pdf_files)} PDF(s) under {base}")

        for pdf_path in pdf_files:
            self._process_one(pdf_path, out)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _process_one(self, pdf_path: Path, out: Path) -> None:
        doc_id = make_doc_id(pdf_path)
        save_path = out / f"{doc_id}_{pdf_path.stem}.json"

        if save_path.exists():
            log.info(f"⏭  Skip (cached): {pdf_path.name}")
            return

        log.info(f"▶  Processing: {pdf_path}")
        try:
            chunks = self.extract_file(pdf_path)
            if not chunks:
                log.warning(f"⚠  No chunks produced for {pdf_path.name}")
                return

            payload = [c.to_dict() for c in chunks]
            save_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(f"✅ {pdf_path.name} → {len(chunks)} chunks → {save_path.name}")

        except Exception as exc:
            log.exception(f"❌ Failed: {pdf_path.name} — {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# 10. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    extractor = PDFExtractor()
    extractor.process_directory("data", output_dir="extracted_chunks")