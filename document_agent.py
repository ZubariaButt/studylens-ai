import os
import re
from typing import Dict, List, Any
from pypdf import PdfReader


class DocumentProcessingAgent:
    """
    Member 2: Document Processing Agent for StudyLens AI.

    Extracts text from uploaded PDFs, cleans whitespace/noise,
    and creates overlapping chunks for downstream agents.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean_text(self, text: str) -> str:
        """Normalize whitespace and unnecessary line breaks."""
        if not text:
            return ""

        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def chunk_text(
        self, text: str, page_number: int = 1
    ) -> List[Dict[str, Any]]:
        """Create overlapping chunks with page metadata."""
        chunks = []
        start = 0
        text_length = len(text)
        chunk_idx = 1
        step = self.chunk_size - self.chunk_overlap

        while start < text_length:
            end = start + self.chunk_size
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunks.append({
                    "chunk_id": chunk_idx,
                    "page": page_number,
                    "text": chunk_content,
                    "char_length": len(chunk_content)
                })

            start += step
            chunk_idx += 1

        return chunks

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point for the Orchestrator.
        Returns the same schema expected by the team workflow.
        """
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found at path: {file_path}",
                "total_pages": 0,
                "total_characters": 0,
                "total_chunks": 0,
                "full_text": "",
                "chunks": []
            }

        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            full_cleaned_text = []
            all_chunks = []

            for page_idx, page in enumerate(reader.pages, start=1):
                raw_page_text = page.extract_text() or ""
                cleaned_page_text = self.clean_text(raw_page_text)

                if cleaned_page_text:
                    full_cleaned_text.append(cleaned_page_text)
                    page_chunks = self.chunk_text(
                        cleaned_page_text,
                        page_number=page_idx
                    )
                    all_chunks.extend(page_chunks)

            combined_text = "\n\n".join(full_cleaned_text)

            for idx, chunk in enumerate(all_chunks, start=1):
                chunk["chunk_id"] = idx

            return {
                "status": "success",
                "file_name": os.path.basename(file_path),
                "total_pages": total_pages,
                "total_characters": len(combined_text),
                "total_chunks": len(all_chunks),
                "full_text": combined_text,
                "chunks": all_chunks
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to parse document: {str(e)}",
                "total_pages": 0,
                "total_characters": 0,
                "total_chunks": 0,
                "full_text": "",
                "chunks": []
            }
