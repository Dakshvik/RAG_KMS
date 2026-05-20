import os
import re
import json
import logging
import hashlib
import gc
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image
import torch

# LangChain Imports
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Native Surya Imports (Bypassing Docling Wrapper)
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

# --- HARDWARE CONSTRAINTS ---
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Final-Universal-Pipeline")

class ProductionDocumentExtractor:
    def __init__(self):
        logger.info("Loading Native Surya Models into RTX 4060 VRAM...")
        # Load the models once at initialization
        self.det_predictor = DetectionPredictor()
        self.rec_predictor = RecognitionPredictor()
        
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Chapter_Module"), ("##", "Section_Topic")],
            strip_headers=False 
        )

    def is_page_corrupted(self, text: str) -> bool:
        """Adaptive, Multi-Lingual Corruption Analyzer."""
        clean_text = text.strip()
        total_chars = len(clean_text)
        
        if total_chars < 30: return False  
            
        unexpected = re.sub(r'[\x00-\x7F\u0C00-\u0C7F\s\W]', '', clean_text)
        if len(unexpected) > (total_chars * 0.05): return True
            
        eng_words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text.lower())
        if len(eng_words) >= 4:
            vowel_less = [w for w in eng_words if not re.search(r'[aeiouy]', w)]
            if len(vowel_less) > (len(eng_words) * 0.4): return True
                
            valid_tokens = {
                "the", "and", "of", "to", "in", "that", "is", "for", "on", "it", "with", "this", "from", "are", "have",
                "management", "leadership", "police", "state", "rules", "court", "officer", "team",
                "chapter", "section", "module", "index", "page", "topics", "contents", "foreword"
            }
            matched = sum(1 for word in eng_words if word in valid_tokens)
            if matched == 0: return True
                
        telugu_chars = re.findall(r'[\u0C00-\u0C7F]', clean_text)
        if total_chars > 100 and len(eng_words) < 4 and len(telugu_chars) < (total_chars * 0.05):
            numbers = re.findall(r'[0-9]', clean_text)
            if len(numbers) < (total_chars * 0.25): return True
                
        return False

    def clean_and_structure_text(self, markdown_text: str) -> str:
        text = re.sub(r'Trg-Pol\.Vach-1-2025/T\.S\.Police/Hyd\s*\d*', '', markdown_text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'^(CHAPTER\s*-\s*[IVXLCDM]+.*)', r'# \1', text, flags=re.MULTILINE)
        text = re.sub(r'^(Sec\.\s*\d+\..*)', r'## \1', text, flags=re.MULTILINE)
        text = re.sub(r'^(MODULE\s+[IVXLCDM]+.*)', r'# \1', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^(\d+\.\s+[A-Z\s]{4,})', r'## \1', text, flags=re.MULTILINE)
        text = re.sub(r'^([A-Z\s]{5,}\s+COMPONENT.*)', r'## \1', text, flags=re.IGNORECASE | re.MULTILINE)
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def generate_doc_id(self, pdf_path: Path) -> str:
        key = f"{pdf_path.parent.name}::{pdf_path.name}".lower()
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def process_pdf_safely(self, pdf_path: Path) -> List[Document]:
        logger.info(f"Processing target document: {pdf_path.name}")
        full_markdown_pages = []
        doc = fitz.open(pdf_path)
        
        language = "telugu" if "telugu" in pdf_path.parts else "english"
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                raw_text = page.get_text("text")
                
                is_corrupted = self.is_page_corrupted(raw_text)
                needs_ocr = (language == "telugu") or is_corrupted
                
                if needs_ocr:
                    reason = "Telugu Document" if language == "telugu" else "Corrupted Unicode"
                    logger.warning(f"Routing Page {page_num + 1} to Native Surya GPU. Reason: {reason}...")
                    
                    try:
                        # Direct Memory Transfer: PyMuPDF to PIL (No temporary file required!)
                        pix = page.get_pixmap(dpi=150)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # Native Surya Inference
                        det_preds = self.det_predictor([img])
                        rec_preds = self.rec_predictor([img], det_preds, languages=[["te", "en"]])
                        
                        # Extract the recognized text payload
                        lines = [line.text for line in rec_preds[0].text_lines]
                        full_markdown_pages.append("\n".join(lines))
                        
                    except Exception as inner_err:
                        logger.error(f"Surya OCR failure on Page {page_num + 1}: {inner_err}")
                    finally:
                        # Aggressive VRAM cleanup to protect your 8GB card
                        if 'img' in locals(): del img
                        if 'det_preds' in locals(): del det_preds
                        if 'rec_preds' in locals(): del rec_preds
                        torch.cuda.empty_cache()
                        gc.collect()
                else:
                    # Fast track CPU extraction for clean English pages
                    try:
                        page_md = page.get_text("markdown")
                    except Exception:
                        page_md = raw_text
                    full_markdown_pages.append(page_md)
                    
            combined_markdown = "\n\n".join(full_markdown_pages)
            structured_markdown = self.clean_and_structure_text(combined_markdown)
            md_docs = self.markdown_splitter.split_text(structured_markdown)
            
            langchain_docs = []
            doc_id = self.generate_doc_id(pdf_path)
            subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else "unknown"
            
            for idx, chunk in enumerate(md_docs):
                if len(chunk.page_content.strip()) < 15: 
                    continue

                metadata = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}-{idx:04d}",
                    "source": str(pdf_path).replace("\\", "/"),
                    "filename": pdf_path.name,
                    "subject": subject,
                    "language": language,
                    "chapter": chunk.metadata.get("Chapter_Module", "Unknown Chapter/Module"),
                    "section": chunk.metadata.get("Section_Topic", "Unknown Section/Topic"),
                    "extraction_method": "native_surya_ocr" if needs_ocr else "pymupdf_fast",
                }
                langchain_docs.append(Document(page_content=chunk.page_content, metadata=metadata))
            
            return langchain_docs
            
        except Exception as e:
            logger.error(f"Critical error processing {pdf_path.name}: {e}")
            return []
        finally:
            doc.close()
            gc.collect()

    def execution_loop(self, input_dir: str, output_dir: str):
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(in_path.rglob("*.pdf"))
        logger.info(f"Targeting batch containing {len(pdf_files)} PDFs.")
        
        for pdf in pdf_files:
            output_file = out_path / f"{pdf.stem}_chunks.json"
            
            if output_file.exists():
                logger.info(f"Skipping {pdf.name} (Already processed)")
                continue
                
            chunks = self.process_pdf_safely(pdf)
            
            if chunks:
                serialized_chunks = [
                    {
                        "id": None,
                        "metadata": chunk.metadata,
                        "page_content": chunk.page_content,
                        "type": "Document"
                    } for chunk in chunks
                ]
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(serialized_chunks, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved {len(chunks)} chunks to {output_file.name}")

if __name__ == "__main__":
    extractor = ProductionDocumentExtractor()
    extractor.execution_loop(input_dir="data", output_dir="extracted_chunks")