import os
import re
import json
import time
import logging
import unicodedata
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- NVIDIA CUDA & MULTI-CORE CPU OPTIMIZATIONS ---
os.environ["TORCH_DEVICE"] = "cuda"
# Docling and Marker will leverage these for underlying linear algebra libraries on your 20-core CPU
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["OPENBLAS_NUM_THREADS"] = "12"

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Docling Imports
from docling.document_converter import DocumentConverter
from docling.chunking import HierarchicalChunker

# --- MARKER V1.10+ IMPORTS ---
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# CONFIGURATION & LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PDFExtraction-RTX4060")

def generate_doc_id(pdf_path: Path) -> str:
    subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
    key = f"{subject}::{pdf_path.name}".lower()
    return hashlib.md5(key.encode()).hexdigest()[:12]

def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}-{str(chunk_index).zfill(4)}"

# --- TEXT CLEANING ---

LIGATURE_MAP = str.maketrans({
    '\ufb00': 'ff', '\ufb01': 'fi', '\ufb02': 'fl', '\ufb03': 'ffi', '\ufb04': 'ffl',
    '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-', '\u00ad': '', '\u00a0': ' ',    
})

def clean_pdf_text(text: str, language: str = "english") -> str:
    if not isinstance(text, str): return "Unknown"
    
    # Remove private use Unicode (often broken PDF fonts)
    text = re.sub(r'[\uE000-\uF8FF]', '', text)
    text = text.translate(LIGATURE_MAP)
    
    # Fix hyphenated word breaks
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Remove isolated page numbers
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    
    # ONLY strip non-ASCII characters if the document is strictly English
    if language == "english":
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", errors="ignore").decode("ascii")
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_text_corrupted(text: str, threshold: float = 0.10) -> bool:
    if not text or len(text.strip()) < 50:
        return False
    
    stop_words = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "it"}
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if len(words) < 15: return False
        
    stop_word_count = sum(1 for word in words if word in stop_words)
    return (stop_word_count / len(words)) < threshold


# --- MAIN EXTRACTION CLASS ---

class ProductionPDFExtractor:
    def __init__(self):
        # 1. Fast Path: Docling (Automatically targets CUDA on Windows if available)
        logger.info("Initializing Docling models (CUDA / CPU Auto-detection)...")
        self.doc_converter = DocumentConverter()
        self.doc_chunker = HierarchicalChunker()
        
        # 2. Slow Path: Marker (Accelerated via NVIDIA CUDA)
        logger.info("Initializing Marker models (NVIDIA CUDA GPU Accelerated)...")
        artifact_dict = create_model_dict()
        self.marker_converter = PdfConverter(artifact_dict=artifact_dict)
        
        # 3. Fallback Splitter (For Marker Markdown)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    def process_fast_path(self, pdf_path: Path) -> List[Document]:
        logger.info(f"⚡ Fast Path (Docling): {pdf_path.name}")
        doc = self.doc_converter.convert(str(pdf_path)).document
        docling_chunks = list(self.doc_chunker.chunk(doc))
        
        # Gibberish Check
        sample_text = " ".join([c.text for c in docling_chunks[:5]])
        if is_text_corrupted(sample_text):
            logger.warning(f"🚨 Corrupted font detected in {pdf_path.name}. Rerouting to Marker (CUDA).")
            return self.process_slow_path(pdf_path, language="english", method="marker_fallback")

        langchain_docs = []
        doc_id = generate_doc_id(pdf_path)

        for idx, chunk in enumerate(docling_chunks):
            clean_text = clean_pdf_text(chunk.text, language="english")
            if not clean_text.strip(): continue

            meta = chunk.meta
            raw_heading = meta.headings[0] if meta and meta.headings else "Unknown"
            
            page_no = 1
            if meta and meta.doc_items and meta.doc_items[0].prov and len(meta.doc_items[0].prov) > 0:
                page_no = meta.doc_items[0].prov[0].page_no

            metadata = {
                "doc_id": doc_id,
                "chunk_id": generate_chunk_id(doc_id, idx),
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "subject": pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem,
                "language": "english",
                "extraction_method": "docling_fast",
                "heading": raw_heading,
                "page": page_no,
            }
            langchain_docs.append(Document(page_content=clean_text, metadata=metadata))
            
        return langchain_docs

    def process_slow_path(self, pdf_path: Path, language: str, method: str) -> List[Document]:
        logger.info(f"🏎️ GPU Path (Marker Vision via CUDA): {pdf_path.name}")
        
        try:
            rendered = self.marker_converter(str(pdf_path))
            text, _, _ = text_from_rendered(rendered)
        except Exception as e:
            logger.error(f"Marker failed on {pdf_path.name}: {e}")
            return []

        if not text.strip(): return []

        md_docs = self.markdown_splitter.split_text(text)
        
        langchain_docs = []
        doc_id = generate_doc_id(pdf_path)
        
        for idx, doc in enumerate(md_docs):
            clean_text = clean_pdf_text(doc.page_content, language=language)
            if not clean_text.strip(): continue

            heading = doc.metadata.get("Header 3") or doc.metadata.get("Header 2") or doc.metadata.get("Header 1") or "Unknown"

            metadata = {
                "doc_id": doc_id,
                "chunk_id": generate_chunk_id(doc_id, idx),
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "subject": pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem,
                "language": language,
                "extraction_method": method,
                "heading": heading,
                "page": "Vision_Extracted" 
            }
            langchain_docs.append(Document(page_content=clean_text, metadata=metadata))

        return langchain_docs

    def extract_file(self, pdf_path_str: str, language: str) -> List[Document]:
        pdf_path = Path(pdf_path_str)
        if not pdf_path.exists(): raise FileNotFoundError(f"File not found: {pdf_path_str}")

        lang = language.strip().lower()
        if lang in ['en', 'english']:
            return self.process_fast_path(pdf_path)
        elif lang in ['te', 'telugu']:
            return self.process_slow_path(pdf_path, language="telugu", method="marker_vision")
        else:
            logger.warning(f"Unsupported language '{language}'. Defaulting to Fast Path.")
            return self.process_fast_path(pdf_path)

    def process_directory_safely(self, data_dir: str, output_dir: str = "extracted_chunks"):
        base_path = Path(data_dir)
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True)
        
        sorted_pdfs = sorted(base_path.rglob("*.pdf"))
        for pdf_path in sorted_pdfs:
            detected_lang = "english" if "english" in pdf_path.parts else "telugu" if "telugu" in pdf_path.parts else "english"
                
            doc_id = generate_doc_id(pdf_path)
            save_file = out_path / f"{doc_id}_{pdf_path.stem}.json"
            
            if save_file.exists():
                logger.info(f"⏭️ Skipping {pdf_path.name}, already extracted.")
                continue

            try:
                docs = self.extract_file(str(pdf_path), language=detected_lang)
                if not docs:
                    continue
                
                docs_dict = [doc.model_dump() for doc in docs]
                
                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump(docs_dict, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"✅ Saved {len(docs)} chunks to {save_file.name}")

            except Exception as e:
                logger.error(f"❌ Failed processing {pdf_path.name}: {str(e)}\n")

if __name__ == "__main__":
    extractor = ProductionPDFExtractor()
    extractor.process_directory_safely("data", output_dir="extracted_chunks")