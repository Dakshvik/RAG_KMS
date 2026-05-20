"""
Production PDF Extraction Pipeline v4.2 — FINAL STABLE VERSION
==============================================================

Optimized for:
- RTX 4060 8GB
- Long-running extraction
- Large multilingual PDF datasets
- VRAM-safe processing
- Resume support
- Corruption recovery

KEY FEATURES:
- True VRAM cleanup
- Skips already processed PDFs
- Removes corrupted outputs
- Streaming Marker extraction
- inference_mode enabled
- CPU Docling
- Automatic CUDA recovery
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

from pathlib import Path
from typing import List, Dict, Tuple
from contextlib import contextmanager

# =============================================================================
# ENVIRONMENT CONFIG
# =============================================================================

os.environ["TORCH_DEVICE"] = "cuda"
os.environ["TORCH_DTYPE"] = "fp16"

# Better for 8GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,max_split_size_mb:64"
)

# Force Docling to CPU
os.environ["DOCLING_DEVICE"] = "cpu"

# OCR batch settings
os.environ["SURYA_DET_BATCH_SIZE"] = "1"
os.environ["SURYA_REC_BATCH_SIZE"] = "1"
os.environ["TEXIFY_BATCH_SIZE"] = "1"

# CPU optimization
_cpu_cores = os.cpu_count() or 1
_thread_count = str(min(12, max(1, _cpu_cores - 2)))

os.environ["OMP_NUM_THREADS"] = _thread_count
os.environ["MKL_NUM_THREADS"] = _thread_count

# =============================================================================
# IMPORTS
# =============================================================================

import torch

torch.set_grad_enabled(False)

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

import pypdf

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("PDFExtractor")

# =============================================================================
# CONSTANTS
# =============================================================================

MIN_CHUNK_CHARS = 40

MARKER_BATCH_PAGES = 4

HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

LIGATURE_MAP = str.maketrans({
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00ad": "",
    "\u00a0": " ",
})

# =============================================================================
# CUDA HELPERS
# =============================================================================

def cuda_available() -> bool:
    return torch.cuda.is_available()


def hard_cuda_reset():

    gc.collect()

    if torch.cuda.is_available():

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


def free_cuda_memory():

    gc.collect()

    if cuda_available():

        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_vram_stats() -> Tuple[float, float, float]:

    if not cuda_available():
        return (0.0, 0.0, 100.0)

    free_bytes, total_bytes = torch.cuda.mem_get_info()

    used_bytes = total_bytes - free_bytes

    used_gb = used_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)

    free_percent = (free_bytes / total_bytes) * 100

    return used_gb, total_gb, free_percent


def log_vram(label=""):

    used, total, free = get_vram_stats()

    logger.info(
        "%s VRAM %.2f / %.2f GB (%.1f%% free)",
        label,
        used,
        total,
        free
    )


class CudaOOMError(RuntimeError):
    pass


@contextmanager
def cuda_memory_guard(label=""):

    try:
        yield

    except torch.cuda.OutOfMemoryError as exc:

        hard_cuda_reset()

        raise CudaOOMError(
            f"CUDA OOM during {label}"
        ) from exc

    finally:

        free_cuda_memory()

# =============================================================================
# TEXT HELPERS
# =============================================================================

def clean_pdf_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = re.sub(r"[\uE000-\uF8FF]", "", text)

    text = text.translate(LIGATURE_MAP)

    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def generate_doc_id(pdf_path: Path) -> str:

    key = str(pdf_path).lower()

    return hashlib.md5(
        key.encode()
    ).hexdigest()[:12]


def generate_chunk_id(doc_id: str, idx: int):

    return f"{doc_id}-{idx:04d}"


def doc_to_dict(doc: Document):

    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

# =============================================================================
# PDF HELPERS
# =============================================================================

def get_pdf_page_count(pdf_path: str):

    with open(pdf_path, "rb") as fh:

        reader = pypdf.PdfReader(fh)

        return len(reader.pages)


def extract_pdf_page_range(
    pdf_path,
    output_pdf,
    page_start,
    page_end
):

    with open(pdf_path, "rb") as fh:

        reader = pypdf.PdfReader(fh)

        writer = pypdf.PdfWriter()

        for i in range(page_start, page_end):

            writer.add_page(reader.pages[i])

        with open(output_pdf, "wb") as out_fh:

            writer.write(out_fh)

# =============================================================================
# MODEL MANAGER
# =============================================================================

class ModelManager:

    def __init__(self):

        self._docling_converter = None
        self._docling_chunker = None

        self._marker_converter = None
        self._marker_artifacts = None

        self._md_splitter = MarkdownHeaderTextSplitter(
            HEADERS_TO_SPLIT
        )

    @property
    def docling_converter(self):

        if self._docling_converter is None:

            logger.info("Loading Docling...")

            self._docling_converter = DocumentConverter()

        return self._docling_converter

    @property
    def docling_chunker(self):

        if self._docling_chunker is None:

            self._docling_chunker = HierarchicalChunker()

        return self._docling_chunker

    @property
    def marker_converter(self):

        if self._marker_converter is None:

            logger.info("Loading Marker models...")

            with cuda_memory_guard("marker_load"):

                self._marker_artifacts = create_model_dict()

                self._marker_converter = PdfConverter(
                    artifact_dict=self._marker_artifacts
                )

        return self._marker_converter

    @property
    def md_splitter(self):

        return self._md_splitter

    def teardown_marker(self):

        logger.info("Destroying Marker models...")

        self._marker_converter = None
        self._marker_artifacts = None

        gc.collect()

        hard_cuda_reset()

# =============================================================================
# MAIN EXTRACTOR
# =============================================================================

class ProductionPDFExtractor:

    def __init__(self):

        self.models = ModelManager()

    # =========================================================================
    # FAST PATH
    # =========================================================================

    def fast_extract(self, pdf_path: Path):

        logger.info("Fast path: %s", pdf_path.name)

        result = self.models.docling_converter.convert(
            str(pdf_path)
        )

        chunks = list(
            self.models.docling_chunker.chunk(
                result.document
            )
        )

        documents = []

        doc_id = generate_doc_id(pdf_path)

        for idx, chunk in enumerate(chunks):

            clean = clean_pdf_text(chunk.text)

            if len(clean) < MIN_CHUNK_CHARS:
                continue

            documents.append(
                Document(
                    page_content=clean,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": generate_chunk_id(doc_id, idx),
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "method": "docling"
                    }
                )
            )

        return documents

    # =========================================================================
    # MARKER BATCH
    # =========================================================================

    def process_marker_batch(
        self,
        batch_pdf: Path,
        doc_id: str,
        start_idx: int
    ):

        try:

            with cuda_memory_guard("marker_batch"):

                with torch.inference_mode():

                    rendered = self.models.marker_converter(
                        str(batch_pdf)
                    )

                    result = text_from_rendered(rendered)

                    text = (
                        result[0]
                        if isinstance(result, (list, tuple))
                        else result
                    )

                del rendered
                del result

                gc.collect()

        except Exception as exc:

            logger.error("Marker failed: %s", exc)

            return []

        if not text:
            return []

        md_docs = self.models.md_splitter.split_text(text)

        del text

        gc.collect()

        documents = []

        for idx, md_doc in enumerate(md_docs):

            clean = clean_pdf_text(
                md_doc.page_content
            )

            if len(clean) < MIN_CHUNK_CHARS:
                continue

            documents.append(
                Document(
                    page_content=clean,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": generate_chunk_id(
                            doc_id,
                            start_idx + idx
                        ),
                        "source": str(batch_pdf),
                        "method": "marker"
                    }
                )
            )

        return documents

    # =========================================================================
    # STREAMING EXTRACTION
    # =========================================================================

    def streaming_extract(
        self,
        pdf_path: Path,
        output_file: str
    ):

        total_pages = get_pdf_page_count(
            str(pdf_path)
        )

        logger.info(
            "Streaming %d pages...",
            total_pages
        )

        doc_id = generate_doc_id(pdf_path)

        chunk_counter = 0

        output_path = Path(output_file)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_dir = tempfile.gettempdir()

        for page_start in range(
            0,
            total_pages,
            MARKER_BATCH_PAGES
        ):

            page_end = min(
                page_start + MARKER_BATCH_PAGES,
                total_pages
            )

            log_vram(f"Before batch {page_start}")

            temp_pdf = (
                Path(temp_dir)
                / f"batch_{page_start}.pdf"
            )

            extract_pdf_page_range(
                str(pdf_path),
                str(temp_pdf),
                page_start,
                page_end
            )

            batch_docs = self.process_marker_batch(
                temp_pdf,
                doc_id,
                chunk_counter
            )

            chunk_counter += len(batch_docs)

            with open(
                output_path,
                "a",
                encoding="utf-8"
            ) as fh:

                for doc in batch_docs:

                    fh.write(
                        json.dumps(
                            doc_to_dict(doc),
                            ensure_ascii=False
                        ) + "\n"
                    )

            del batch_docs

            gc.collect()

            temp_pdf.unlink(missing_ok=True)

            hard_cuda_reset()

            log_vram(f"After batch {page_start}")

        return chunk_counter

    # =========================================================================
    # SINGLE PDF
    # =========================================================================

    def process_pdf(
        self,
        pdf_path: Path,
        output_file: str,
        language="english"
    ):

        logger.info("=" * 70)
        logger.info("Processing %s", pdf_path.name)

        try:

            if language == "english":

                docs = self.fast_extract(pdf_path)

                with open(
                    output_file,
                    "w",
                    encoding="utf-8"
                ) as fh:

                    json.dump(
                        [doc_to_dict(d) for d in docs],
                        fh,
                        ensure_ascii=False,
                        indent=2
                    )

                count = len(docs)

                del docs

            else:

                count = self.streaming_extract(
                    pdf_path,
                    output_file
                )

            logger.info(
                "Finished %s -> %d chunks",
                pdf_path.name,
                count
            )

            return count

        finally:

            logger.info("FULL VRAM RESET")

            self.models.teardown_marker()

            hard_cuda_reset()

            time.sleep(2)

            log_vram("Post-cleanup")

    # =========================================================================
    # DIRECTORY PROCESSING
    # =========================================================================

    def process_directory(
        self,
        data_dir,
        output_dir="extracted_chunks",
        force=False
    ):

        base = Path(data_dir)

        out = Path(output_dir)

        out.mkdir(
            parents=True,
            exist_ok=True
        )

        pdfs = sorted(base.rglob("*.pdf"))

        logger.info(
            "Found %d PDFs",
            len(pdfs)
        )

        stats = {
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "chunks": 0
        }

        for pdf in pdfs:

            try:

                lang = (
                    "telugu"
                    if "telugu" in str(pdf).lower()
                    else "english"
                )

                doc_id = generate_doc_id(pdf)

                save_file = (
                    out /
                    f"{doc_id}_{pdf.stem}.json"
                )

                # =============================================================
                # SKIP VALID FILES
                # =============================================================

                if (
                    not force
                    and save_file.exists()
                    and save_file.stat().st_size > 1000
                ):

                    logger.info(
                        "Skipping already processed PDF: %s",
                        pdf.name
                    )

                    stats["skipped"] += 1

                    continue

                # =============================================================
                # REMOVE CORRUPTED FILES
                # =============================================================

                if save_file.exists():

                    try:

                        if save_file.stat().st_size < 1000:

                            logger.warning(
                                "Removing corrupted output: %s",
                                save_file.name
                            )

                            save_file.unlink()

                    except Exception:
                        pass

                logger.info("=" * 70)

                logger.info(
                    "Processing %s [lang=%s]",
                    pdf.name,
                    lang
                )

                t0 = time.perf_counter()

                chunks = self.process_pdf(
                    pdf,
                    str(save_file),
                    lang
                )

                elapsed = time.perf_counter() - t0

                stats["processed"] += 1
                stats["chunks"] += chunks

                logger.info(
                    "SUCCESS %s -> %d chunks [%.1fs]",
                    pdf.name,
                    chunks,
                    elapsed
                )

                gc.collect()

                hard_cuda_reset()

                time.sleep(1)

            except Exception as exc:

                logger.error(
                    "FAILED %s: %s",
                    pdf.name,
                    exc
                )

                traceback.print_exc()

                stats["failed"] += 1

                hard_cuda_reset()

                time.sleep(3)

        logger.info("=" * 70)

        logger.info(
            "FINAL SUMMARY | processed=%d skipped=%d failed=%d chunks=%d",
            stats["processed"],
            stats["skipped"],
            stats["failed"],
            stats["chunks"]
        )

        return stats

# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("PDF Extraction Pipeline v4.2 FINAL")
    logger.info("=" * 70)

    log_vram("Initial")

    extractor = ProductionPDFExtractor()

    summary = extractor.process_directory(
        data_dir="data",
        output_dir="extracted_chunks",
        force=False
    )

    logger.info("=" * 70)

    log_vram("Final")

    print(json.dumps(summary, indent=2))