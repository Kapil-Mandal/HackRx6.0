# app/utils.py
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import docx
from typing import List

def get_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = pypdf.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def get_text_from_docx(docx_path: str) -> str:
    """Extracts text from a DOCX file."""
    # This is a placeholder implementation. You would use the python-docx library here.
    print("DOCX processing not fully implemented.")
    return ""

def get_text_chunks(text: str) -> List[str]:
    """Splits a long text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks