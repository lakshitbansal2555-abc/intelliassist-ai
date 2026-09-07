"""
document_loader.py
===================
Handles the "Document Upload (PDF/TXT/DOCX)" feature. Two jobs:

1. extract_text()   - turn an uploaded file into plain text.
2. chunk_documents() - split that text into overlapping chunks (LangChain
   Document objects) small enough to embed and retrieve accurately.

Keeping this isolated from the rest of the app means adding a new file
type later (e.g. .pptx) only touches this one file.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import docx2txt
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@dataclass
class LoadedDocument:
    filename: str
    text: str
    num_pages: int | None = None


def extract_text(uploaded_file) -> LoadedDocument:
    """
    uploaded_file: a Streamlit UploadedFile (has .name and behaves like a
    file object). Returns the extracted plain text plus a page count when
    available (PDFs only - useful for source citations).
    """
    name = uploaded_file.name
    suffix = name.lower().rsplit(".", 1)[-1]

    if suffix == "pdf":
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        return LoadedDocument(filename=name, text=text, num_pages=len(reader.pages))

    if suffix == "docx":
        # docx2txt wants a path or file-like object; BytesIO works directly.
        text = docx2txt.process(io.BytesIO(uploaded_file.getvalue()))
        return LoadedDocument(filename=name, text=text)

    if suffix == "txt":
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        return LoadedDocument(filename=name, text=text)

    raise ValueError(f"Unsupported file type: .{suffix}. Use PDF, DOCX, or TXT.")


def chunk_documents(
    loaded_docs: list[LoadedDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Splits every loaded document into overlapping chunks and tags each
    chunk with metadata (source filename + chunk index) so the chatbot
    can cite exactly where an answer came from later.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in loaded_docs:
        if not doc.text.strip():
            continue
        pieces = splitter.split_text(doc.text)
        for i, piece in enumerate(pieces):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "source": doc.filename,
                        "chunk_id": i,
                    },
                )
            )
    return chunks
