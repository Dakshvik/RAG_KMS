import os
import re
import json
import logging
import hashlib
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Tuple, Any

from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- WINDOWS PATH CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\poppler\Library\bin' 

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Parallel-Tesseract-Pipeline")

os.environ["OMP_NUM_THREADS"] = "1"

def generate_doc_id(pdf_path: Path) -> str:
    subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
    key = f"{subject}::{pdf_path.name}".lower()
    return hashlib.md5(key.encode()).hexdigest()[:12]

def clean_ocr_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ocr_page_worker(page_data: Tuple[int, Any]) -> Tuple[int, str]:
    page_num, image = page_data
    try:
        text = pytesseract.image_to_string(image, lang="tel+eng")
        return page_num, clean_ocr_text(text)
    except Exception as e:
        return page_num, f"[OCR_ERROR: {str(e)}]"

class ParallelOCRPipeline:
    def __init__(self, dpi: int = 200, workers: int = None):
        self.dpi = dpi
        self.workers = workers or max(1, (os.cpu_count() or 4) - 1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", "।", ".", " ", ""]
        )

    def extract_and_chunk(self, pdf_path: Path) -> List[Document]:
        images = convert_from_path(str(pdf_path), dpi=self.dpi, poppler_path=POPPLER_PATH)
        page_data_list = [(i + 1, img) for i, img in enumerate(images)]
        extracted_pages = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            for page_num, text in executor.map(ocr_page_worker, page_data_list):
                extracted_pages[page_num] = text
        
        return self.create_chunks(pdf_path, extracted_pages)

    def create_chunks(self, pdf_path: Path, extracted_pages: Dict[int, str]) -> List[Document]:
        doc_id = generate_doc_id(pdf_path)
        subject = pdf_path.parts[-2] if len(pdf_path.parts) > 1 else pdf_path.stem
        all_chunks = []
        for page_num, text in extracted_pages.items():
            if not text.strip(): continue
            for idx, chunk_text in enumerate(self.text_splitter.split_text(text)):
                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "doc_id": doc_id, "chunk_id": f"{doc_id}-p{page_num}-{idx:03d}",
                        "source": str(pdf_path).replace("\\", "/"), "page": page_num,
                        "subject": subject, "language": "tel+eng"
                    }
                ))
        return all_chunks

    def process_directory(self, data_dir: str, output_dir: str):
        base_path, out_path = Path(data_dir), Path(output_dir)
        out_path.mkdir(exist_ok=True)
        for pdf_path in sorted(base_path.rglob("*.pdf")):
            save_file = out_path / f"{generate_doc_id(pdf_path)}_{pdf_path.stem}.json"
            if save_file.exists(): continue
            docs = self.extract_and_chunk(pdf_path)
            if docs:
                with open(save_file, "w", encoding="utf-8") as f:
                    json.dump([d.model_dump() for d in docs], f, ensure_ascii=False, indent=2)
                logger.info(f"✅ Processed {pdf_path.name}")

if __name__ == "__main__":
    # WRAP IN MAIN TO PREVENT MULTIPROCESSING CRASHES
    extractor = ParallelOCRPipeline(dpi=200)
    extractor.process_directory(data_dir="data", output_dir="extracted_chunks_1")