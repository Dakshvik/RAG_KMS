import os
import re
import json
import time
import logging
import unicodedata
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any

# LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Docling (English)
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

# OCR (Telugu & corrupted English)
import pytesseract
from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path   # <-- for lazy page count


# ── CONFIGURATION & LOGGING ──────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PDFExtraction")


class Config:
    # OCR settings (adjust paths for your system)
    # TESSERACT_CMD = '/opt/homebrew/bin/tesseract'      # Mac (Homebrew)
    # POPPLER_PATH  = '/opt/homebrew/bin'
    # Windows examples (uncomment & edit):
    TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH  = r'C:\poppler\Library\bin'

    OCR_DPI      = 150
    OCR_WORKERS  = 10         

    # Text splitter settings
    CHUNK_SIZE    = 1000
    CHUNK_OVERLAP = 150

# Set tesseract command globally
pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD


# ── UTILITY FUNCTIONS ────────────────────────────────────────────
def generate_doc_id(pdf_path: Path) -> str:
    """Stable ID based on folder + filename."""
    subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
    key = f"{subject}::{pdf_path.name}".lower()
    return hashlib.md5(key.encode()).hexdigest()[:12]


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Unique chunk ID: <doc_id>-<zero‑padded index>"""
    return f"{doc_id}-{str(chunk_index).zfill(4)}"


LIGATURE_MAP = str.maketrans({
    '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl', '\ufb03': 'ffi', '\ufb04': 'ffl',
    '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-', '\u00ad': '', '\u00a0': ' ',
})


def clean_pdf_text_en(text: str) -> str:
    """Clean English text: ligatures, line‑breaks, ASCII normalisation."""
    if not isinstance(text, str):
        return "Unknown"
    text = re.sub(r'[\uE000-\uF8FF]', '', text)          # remove PUA characters
    text = text.translate(LIGATURE_MAP)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)       # hyphenated line break
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def clean_heading_text_en(text: str) -> str:
    """Remove leading numbers / bullet points from a heading."""
    if not isinstance(text, str):
        return "unknown"
    text = clean_pdf_text_en(text)
    text = re.sub(r'^([a-z]|[ivxlcdm]+|\d+)(\.\d+)*\.\s*', '', text)
    text = re.sub(r'^[-•–―]\s*', '', text)
    return text.strip()


def is_text_corrupted(text: str, threshold: float = 0.10) -> bool:
    """
    Detect font‑encoding corruption by checking common English stop‑word frequency.
    If too few stop‑words appear, the text is likely garbled → OCR fallback needed.
    """
    if not text or len(text.strip()) < 50:
        return False

    stop_words = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
        "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
        "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
        "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when"
    }
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if len(words) < 15:
        return False
    stop_word_count = sum(1 for word in words if word in stop_words)
    return (stop_word_count / len(words)) < threshold


# ── MULTIPROCESSING OCR WORKER (top‑level for pickle) ───────────
def _process_single_page_ocr(args: tuple) -> Dict[str, Any]:
    """Run Tesseract on a single image. Called by ProcessPoolExecutor."""
    page_img, page_num, lang_code = args
    try:
        text = pytesseract.image_to_string(page_img, lang=lang_code, config='--psm 1')
        return {"page": page_num, "content": text.strip()}
    except Exception as e:
        return {"page": page_num, "content": "", "error": str(e)}


# ── MAIN EXTRACTION CLASS ───────────────────────────────────────
class AdvancedPDFExtractor:
    def __init__(self):
        # Docling for clean English
        self.doc_converter = DocumentConverter()
        self.doc_chunker = HierarchicalChunker()

        # OCR splitter for Telugu / corrupted English
        self.ocr_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    # ── English path (Docling) ──────────────────────────────────
    def process_english_pdf(self, pdf_path: Path) -> List[Document]:
        logger.info(f"Starting Docling extraction for: {pdf_path.name}")
        doc = self.doc_converter.convert(str(pdf_path)).document
        docling_chunks = list(self.doc_chunker.chunk(doc))

        # Quick corruption check using the first few chunks
        sample_text = " ".join([c.text for c in docling_chunks[:5]])
        if is_text_corrupted(sample_text):
            logger.warning(f"🚨 Corrupted font detected in {pdf_path.name}. Rerouting to OCR Fallback.")
            return self.process_ocr_pipeline(pdf_path, language="english",
                                             ocr_lang="eng", method="ocr_fallback")

        langchain_docs = []
        doc_id = generate_doc_id(pdf_path)

        for idx, chunk in enumerate(docling_chunks):
            clean_text = clean_pdf_text_en(chunk.text)
            if not clean_text.strip():
                continue

            meta = chunk.meta
            raw_heading = meta.headings[0] if meta and meta.headings else "Unknown"

            # Page number from Docling provenance (if available)
            page_no = 1
            if (meta and meta.doc_items and
                meta.doc_items[0].prov and
                len(meta.doc_items[0].prov) > 0):
                page_no = meta.doc_items[0].prov[0].page_no

            metadata = {
                "doc_id": doc_id,
                "chunk_id": generate_chunk_id(doc_id, idx),
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "subject": pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem,
                "language": "english",
                "extraction_method": "docling",
                "heading": clean_heading_text_en(raw_heading),
                "page": page_no,
            }
            langchain_docs.append(Document(page_content=clean_text, metadata=metadata))

        return langchain_docs

    # ── OCR path (Telugu + corrupted English) ──────────────────
    def process_ocr_pipeline(self, pdf_path: Path, language: str,
                             ocr_lang: str, method: str) -> List[Document]:
        """
        Memory‑optimised OCR: converts pages ONE AT A TIME using a generator,
        so at most `Config.OCR_WORKERS + 1` images exist in RAM simultaneously.
        """
        logger.info(f"Starting OCR extraction ({ocr_lang}) for: {pdf_path.name}")

        # 1. Cheaply get total number of pages
        info = pdfinfo_from_path(str(pdf_path), poppler_path=Config.POPPLER_PATH)
        total_pages = info["Pages"]

        # 2. Lazy page‑by‑page generator
        def page_iterator():
            for page_num in range(1, total_pages + 1):
                # Convert only this one page
                images = convert_from_path(
                    str(pdf_path),
                    dpi=Config.OCR_DPI,
                    first_page=page_num,
                    last_page=page_num,
                    poppler_path=Config.POPPLER_PATH,
                )
                if images:
                    yield (images[0], page_num, ocr_lang)

        # 3. Map over the generator (lazy pulling)
        with ProcessPoolExecutor(max_workers=Config.OCR_WORKERS) as executor:
            ocr_results = list(executor.map(_process_single_page_ocr, page_iterator()))

        # 4. Sort results by page number
        ocr_results.sort(key=lambda x: x['page'])

        # 5. Chunk and build LangChain Documents
        langchain_docs = []
        doc_id = generate_doc_id(pdf_path)

        for page_data in ocr_results:
            if page_data.get('error'):
                logger.error(f"OCR failed on page {page_data['page']} of {pdf_path.name}: {page_data['error']}")
                continue
            if not page_data['content'].strip():
                continue

            page_chunks = self.ocr_splitter.split_text(page_data['content'])
            for chunk_text in page_chunks:
                local_chunk_idx = len(langchain_docs)
                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": generate_chunk_id(doc_id, local_chunk_idx),
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "subject": pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem,
                    "language": language,
                    "extraction_method": method,
                    "page": page_data['page']
                }
                langchain_docs.append(Document(page_content=chunk_text, metadata=metadata))

        return langchain_docs

    # ── Language router ─────────────────────────────────────────
    def extract_file(self, pdf_path_str: str, language: str) -> List[Document]:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path_str}")

        lang = language.strip().lower()
        if lang in ['en', 'english']:
            return self.process_english_pdf(pdf_path)
        elif lang in ['te', 'telugu']:
            return self.process_ocr_pipeline(pdf_path, language="telugu",
                                             ocr_lang="tel+eng", method="ocr_standard")
        else:
            logger.warning(f"Unsupported language '{language}'. Defaulting to English.")
            return self.process_english_pdf(pdf_path)

    # ── Batch processing with cooldown ──────────────────────────
    def process_directory_safely(self, data_dir: str, output_dir: str = "extracted_chunks"):
        base_path = Path(data_dir)
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True)

        if not base_path.exists() or not base_path.is_dir():
            logger.error(f"Invalid directory: {data_dir}")
            return

        sorted_pdfs = sorted(base_path.rglob("*.pdf"))
        for pdf_path in sorted_pdfs:
            # Simple language detection from folder name
            if "english" in pdf_path.parts:
                detected_lang = "english"
            elif "telugu" in pdf_path.parts:
                detected_lang = "telugu"
            else:
                logger.warning(f"Skipping {pdf_path.name} - unknown language from path.")
                continue

            doc_id = generate_doc_id(pdf_path)
            save_file = out_path / f"{doc_id}_{pdf_path.stem}.json"

            if save_file.exists():
                logger.info(f"⏭️ Skipping {pdf_path.name}, already extracted ({save_file.name}).")
                continue

            try:
                logger.info(f"--- Starting fresh extraction for {pdf_path.name} ---")
                docs = self.extract_file(str(pdf_path), language=detected_lang)
                if not docs:
                    continue

                docs_dict = [doc.model_dump() for doc in docs]
                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump(docs_dict, f, ensure_ascii=False, indent=2)

                logger.info(f"✅ Saved {len(docs)} text chunks to {save_file.name}")

                # Gentle cooldown to prevent overheating
                logger.info("❄️ Cooling down CPU for 10 seconds...\n")
                time.sleep(10)

            except Exception as e:
                logger.error(f"❌ Failed processing {pdf_path.name}: {str(e)}\n")


# ── EXECUTION ────────────────────────────────────────────────────
if __name__ == "__main__":
    extractor = AdvancedPDFExtractor()
    target_directory = "data"
    output_directory = "extracted_chunks_1"

    logger.info("Starting Safe Extraction Pipeline. (Press Ctrl+C to pause anytime)")
    extractor.process_directory_safely(target_directory, output_dir=output_directory)
    logger.info("Batch extraction finished successfully.")