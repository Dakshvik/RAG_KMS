"""
Production PDF Extraction Pipeline
===================================
Optimized for RTX 4060 8GB | Windows | Multi-language (English + Telugu)

Architecture
------------
Fast path  (English text PDFs) : pypdfium2  — pure text extraction, zero RAM spike,
                                              zero GPU usage, ~1s per 100 pages.
Slow path  (Telugu / scanned)  : Marker     — GPU vision+OCR, fp16, per-file model
                                              load/destroy to stay within 8 GB VRAM.

Why pypdfium2 instead of Docling for the fast path
---------------------------------------------------
Docling's StandardPdfPipeline rasterizes every page to a high-DPI bitmap for its
DocLayNet layout model.  On a 232-page PDF this allocates ~15-20 GB of RAM
(std::bad_alloc).  pypdfium2 extracts the embedded text stream directly —
no rasterization, no neural models, no RAM or VRAM pressure at all.

v3 changes
----------
- Replaced Docling fast path with pypdfium2 text extraction (fixes bad_alloc)
- Corruption check now uses pypdfium2 character yield ratio (more reliable)
- _no_cuda() guard kept for any residual library side-effects
- All Surya batch env vars present and set before torch import
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
from typing import Generator, List, Optional, Tuple

# ── Environment flags — must precede ALL torch / surya / marker imports ───────

# Windows CUDA allocator: expandable_segments is Linux-only
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Surya (OCR backbone inside Marker) batch sizes
os.environ["SURYA_BATCH_SIZE"]       = "1"
os.environ["RECOGNITION_BATCH_SIZE"] = "16"
os.environ["DETECTOR_BATCH_SIZE"]    = "2"
os.environ["ORDER_BATCH_SIZE"]       = "4"

# CPU thread budget (leave 4 cores for OS / GPU handoffs)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "16")

import torch  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pdf-extractor")


# =============================================================================
# 1. DATA MODELS
# =============================================================================

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
    total_chunks: int
    prev_chunk_id: str = ""
    next_chunk_id: str = ""
    char_count: int = 0


@dataclass
class Chunk:
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        return {"page_content": self.text, "metadata": asdict(self.metadata)}


# =============================================================================
# 2. ID HELPERS
# =============================================================================

def make_doc_id(pdf_path: Path) -> str:
    subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
    key = f"{subject}::{pdf_path.name}".lower()
    return hashlib.md5(key.encode()).hexdigest()[:12]


def make_chunk_id(doc_id: str, index: int) -> str:
    return f"{doc_id}-{index:04d}"


# =============================================================================
# 3. TEXT CLEANING
# =============================================================================

_LIGATURES = str.maketrans({
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl",
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00ad": "", "\u00a0": " ",
})

_BOILERPLATE_RE = re.compile(
    r"(?im)"
    r"(^\s*page\s+\d+\s*(of\s+\d+)?\s*$"
    r"|^\s*\d{1,4}\s*$"
    r"|^\s*(table\s+of\s+contents|references)\s*$"
    r")"
)


def clean_text(text: str, language: str = "english") -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\uE000-\uF8FF]", "", text)
    text = text.translate(_LIGATURES)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = _BOILERPLATE_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if language == "english":
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", errors="ignore").decode("ascii")
    return text.strip()


# =============================================================================
# 4. CHUNKING  (retrieval-quality: target size + sentence overlap)
# =============================================================================

MIN_CHUNK_CHARS    = 200
TARGET_CHUNK_CHARS = 1200
MAX_CHUNK_CHARS    = 2000
OVERLAP_CHARS      = 150


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _merge_to_target(
    paragraphs: List[str],
    target: int = TARGET_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> List[str]:
    chunks: List[str] = []
    buffer = ""
    tail = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        candidate = (f"{tail} {buffer} {para}".strip() if buffer
                     else f"{tail} {para}".strip())

        if len(candidate) >= max_chars:
            if buffer:
                chunks.append(f"{tail} {buffer}".strip())
                tail = buffer[-overlap:] if len(buffer) > overlap else buffer
                buffer = ""
            sub = tail
            for sent in _split_sentences(para):
                if len(sub) + len(sent) + 1 > max_chars:
                    if sub.strip():
                        chunks.append(sub.strip())
                    sub = sub[-overlap:] if len(sub) > overlap else sub
                sub = f"{sub} {sent}".strip()
            buffer = sub
            tail = ""
        elif len(candidate) >= target:
            chunks.append(candidate)
            sents = _split_sentences(candidate)
            tail = sents[-1] if sents and len(sents[-1]) <= overlap else candidate[-overlap:]
            buffer = ""
        else:
            buffer = candidate

    leftover = f"{tail} {buffer}".strip() if buffer else ""
    if leftover:
        chunks.append(leftover)

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]


# =============================================================================
# 5. CONTEXT MANAGERS
# =============================================================================

@contextmanager
def vram_guard(label: str = "") -> Generator[None, None, None]:
    """Flush GPU allocator before/after a block and log peak usage."""
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
            peak_mb     = torch.cuda.max_memory_allocated() / 1024 ** 2
            reserved_mb = torch.cuda.memory_reserved()      / 1024 ** 2
            log.info(
                f"[{label}] VRAM peak={peak_mb:.0f} MB  "
                f"reserved-after-flush={reserved_mb:.0f} MB"
            )


@contextmanager
def _no_cuda() -> Generator[None, None, None]:
    """Temporarily hide CUDA so library code cannot self-migrate to GPU."""
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original


# =============================================================================
# 6. FAST PATH — pypdfium2 pure text extraction
#
#    pypdfium2 ships with Docling so it is already installed.
#    It reads the embedded text stream directly:
#      - Zero rasterization  → zero RAM spike
#      - One page in memory at a time  → O(1) RAM
#      - Zero CUDA / no models at all
#
#    Fallback trigger: if fewer than 10 % of pages yield text the PDF is
#    likely scanned and we raise ValueError → Marker takes over.
# =============================================================================

class PypdfiumStrategy:

    def extract(self, pdf_path: Path, language: str) -> List[Chunk]:
        import pypdfium2 as pdfium   # bundled with Docling

        doc_id  = make_doc_id(pdf_path)
        subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem

        pages: List[Tuple[int, str]] = []
        pdf = pdfium.PdfDocument(str(pdf_path))
        total_pages = len(pdf)

        try:
            for page_idx in range(total_pages):
                page     = pdf[page_idx]
                textpage = page.get_textpage()
                raw      = textpage.get_text_range()
                textpage.close()
                page.close()

                text = clean_text(raw, language)
                if text:
                    pages.append((page_idx + 1, text))   # 1-indexed
        finally:
            pdf.close()

        # Scanned-PDF gate
        if len(pages) < max(1, int(total_pages * 0.10)):
            raise ValueError(
                f"pypdfium2: only {len(pages)}/{total_pages} pages had text "
                f"(scanned PDF) — rerouting to Marker"
            )

        log.info(
            f"  pypdfium2: {len(pages)}/{total_pages} pages extracted "
            f"[{pdf_path.name}]"
        )

        # Build chunks, use first line of each chunk as the heading
        raw_chunks: List[Tuple[str, int, str]] = []
        for page_no, page_text in pages:
            paras = [p.strip() for p in re.split(r"\n\n+", page_text) if p.strip()]
            for merged in _merge_to_target(paras):
                first_line = merged.split("\n", 1)[0][:80]
                raw_chunks.append((first_line, page_no, merged))

        return _finalise(raw_chunks, doc_id, subject, str(pdf_path), language, "pypdfium2_text")


# =============================================================================
# 7. SLOW PATH — Marker GPU vision + OCR
# =============================================================================

class MarkerStrategy:
    """
    Per-file model load + destroy keeps VRAM at baseline between files.
    All sub-models cast to fp16 after loading to prevent scatter() dtype crash.
    """

    @staticmethod
    def _to_fp16(artifact_dict: dict) -> None:
        for name, obj in artifact_dict.items():
            if isinstance(obj, torch.nn.Module):
                try:
                    obj.half()
                    log.debug(f"  Cast {name} -> fp16")
                except Exception as exc:
                    log.warning(f"  Could not cast {name} to fp16: {exc}")

    def extract(self, pdf_path: Path, language: str) -> List[Chunk]:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        doc_id  = make_doc_id(pdf_path)
        subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
        text    = ""

        with vram_guard("marker"):
            artifact_dict = converter = rendered = None
            try:
                log.info(f"  Marker: loading models -> VRAM [{pdf_path.name}]")
                artifact_dict = create_model_dict()
                self._to_fp16(artifact_dict)

                converter  = PdfConverter(artifact_dict=artifact_dict)
                rendered   = converter(str(pdf_path))
                text, _, _ = text_from_rendered(rendered)

            except Exception as exc:
                log.error(f"Marker failed on {pdf_path.name}: {exc}", exc_info=True)
                return []
            finally:
                for obj in (rendered, converter, artifact_dict):
                    if obj is not None:
                        del obj

        if not text.strip():
            return []

        return _markdown_to_chunks(
            text, doc_id, subject, str(pdf_path), language, "marker_vision"
        )


# =============================================================================
# 8. CHUNK ASSEMBLY
# =============================================================================

def _markdown_to_chunks(
    md_text: str, doc_id: str, subject: str,
    source: str, language: str, method: str,
) -> List[Chunk]:
    header_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    positions = [(m.start(), m.group(2).strip()) for m in header_re.finditer(md_text)]
    positions.append((len(md_text), "__END__"))

    raw: list = []

    # Preamble before the first header
    if positions[0][0] > 0:
        pre = clean_text(md_text[: positions[0][0]], language)
        for merged in _merge_to_target(
            [p for p in re.split(r"\n\n+", pre) if p.strip()]
        ):
            raw.append(("Preamble", "vision", merged))

    for i, (start, heading) in enumerate(positions[:-1]):
        body = header_re.sub("", md_text[start: positions[i + 1][0]], count=1).strip()
        body = clean_text(body, language)
        for merged in _merge_to_target(
            [p for p in re.split(r"\n\n+", body) if p.strip()]
        ):
            raw.append((heading, "vision", merged))

    return _finalise(raw, doc_id, subject, source, language, method)


def _finalise(
    raw: list, doc_id: str, subject: str,
    source: str, language: str, method: str,
) -> List[Chunk]:
    filename = Path(source).name
    chunks: List[Chunk] = []

    for idx, (heading, page, text) in enumerate(raw):
        chunks.append(Chunk(
            text=text,
            metadata=ChunkMetadata(
                doc_id=doc_id,
                chunk_id=make_chunk_id(doc_id, idx),
                source=source.replace("\\", "/"),
                filename=filename,
                subject=subject,
                language=language,
                extraction_method=method,
                heading=heading,
                page=page,
                chunk_index=idx,
                total_chunks=len(raw),
                char_count=len(text),
            ),
        ))

    for i, chunk in enumerate(chunks):
        chunk.metadata.total_chunks  = len(chunks)
        chunk.metadata.prev_chunk_id = chunks[i - 1].metadata.chunk_id if i > 0 else ""
        chunk.metadata.next_chunk_id = (
            chunks[i + 1].metadata.chunk_id if i < len(chunks) - 1 else ""
        )

    return chunks


# =============================================================================
# 9. LANGUAGE DETECTOR
# =============================================================================

def detect_language(pdf_path: Path) -> str:
    parts_lower = [p.lower() for p in pdf_path.parts]
    if "telugu" in parts_lower:
        return "telugu"
    return "english"


# =============================================================================
# 10. ORCHESTRATOR
# =============================================================================

class PDFExtractor:
    """
    Routing:
        English PDF  ->  pypdfium2  (0 RAM, 0 VRAM, ~1s/100 pages)
                         if scanned -> Marker (GPU)
        Telugu PDF   ->  Marker (GPU, vision OCR)
    """

    def __init__(self) -> None:
        self._pypdfium = PypdfiumStrategy()
        self._marker   = MarkerStrategy()

    def extract_file(self, pdf_path: Path, language: Optional[str] = None) -> List[Chunk]:
        lang = language or detect_language(pdf_path)
        if lang == "english":
            try:
                return self._pypdfium.extract(pdf_path, lang)
            except ValueError as exc:
                log.warning(f"{pdf_path.name}: {exc} — falling back to Marker")
                return self._marker.extract(pdf_path, lang)
        return self._marker.extract(pdf_path, lang)

    def process_directory(
        self,
        data_dir: str | Path,
        output_dir: str | Path = "extracted_chunks",
    ) -> None:
        base = Path(data_dir)
        out  = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(base.rglob("*.pdf"))
        log.info(f"Found {len(pdf_files)} PDF(s) under {base}")
        for pdf_path in pdf_files:
            self._process_one(pdf_path, out)

    def _process_one(self, pdf_path: Path, out: Path) -> None:
        doc_id    = make_doc_id(pdf_path)
        save_path = out / f"{doc_id}_{pdf_path.stem}.json"

        if save_path.exists():
            log.info(f"Skip (cached): {pdf_path.name}")
            return

        log.info(f"Processing: {pdf_path}")
        try:
            chunks = self.extract_file(pdf_path)
            if not chunks:
                log.warning(f"No chunks produced for {pdf_path.name}")
                return
            save_path.write_text(
                json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(f"{pdf_path.name} -> {len(chunks)} chunks -> {save_path.name}")
        except Exception as exc:
            log.exception(f"Failed: {pdf_path.name} — {exc}")


# =============================================================================
# 11. ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    extractor = PDFExtractor()
    extractor.process_directory("data", output_dir="extracted_chunks")