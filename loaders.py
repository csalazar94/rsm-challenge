import os

from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from unstructured.cleaners.core import clean_extra_whitespace

from logger import logger

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 300
PDF_PROCESSING_MODE = "elements"
PDF_PROCESSING_STRATEGY = "hi_res"


async def get_chunks_from_pdf(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Extract and chunk text content from a PDF file using high-resolution processing.

    This function loads a PDF file, processes it to extract structured content including
    tables, filters metadata, and splits the content into manageable chunks for further
    processing (e.g., embeddings, search indexing).

    Args:
        file_path (str): Path to the PDF file to be processed

    Returns:
        list: List of document chunks with text content and metadata

    Raises:
        HTTPException: 500 status code if PDF processing fails

    Note:
        Uses high-resolution strategy for better text extraction quality and
        infers table structure for improved document understanding.
    """
    try:
        logger.info(f"Starting PDF processing for: {file_path}")

        # TODO: Add support for other file types
        # TODO: Add support for different PDF processing modes and strategies

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}",
            )

        if not file_path.lower().endswith(".pdf"):
            logger.error(f"Unsupported file type: {file_path}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_path}. Only PDF files are supported.",
            )

        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the limit of 50MB: {file_path}",
            )

        loader = UnstructuredPDFLoader(
            file_path,
            mode=PDF_PROCESSING_MODE,
            strategy=PDF_PROCESSING_STRATEGY,
            post_processors=[clean_extra_whitespace],
            infer_table_structure=True,
        )
        logger.debug(
            f"Configured PDF loader with mode={PDF_PROCESSING_MODE}, "
            f"strategy={PDF_PROCESSING_STRATEGY}"
        )

        docs = await loader.aload()
        if not docs or len(docs) == 0:
            logger.warning(f"No documents found in PDF: {file_path}")
            raise HTTPException(
                status_code=400,
                detail=f"No documents found in the PDF file: {file_path}",
            )
        logger.debug(f"Loaded {len(docs)} documents from PDF.")

        filtered_docs = filter_complex_metadata(docs)
        logger.debug("Filtered complex metadata from documents.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        logger.debug(
            f"Configured text splitter: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}"
        )
        chunks = text_splitter.split_documents(filtered_docs)
        logger.info(
            f"Successfully processed PDF: {file_path} -> " f"{len(chunks)} chunks"
        )
    except Exception as e:
        logger.error(f"Failed to process PDF {file_path}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing file: {file_path}",
        )
    return chunks
