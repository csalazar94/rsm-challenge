from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from unstructured.cleaners.core import clean_extra_whitespace

import logger

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 300
PDF_PROCESSING_MODE = "elements"
PDF_PROCESSING_STRATEGY = "hi_res"


async def get_chunks_from_pdf(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
):
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
        # TODO: Add support for other file types
        # TODO: Check if the file exists before processing
        # TODO: Check if the file is a valid PDF
        # TODO: Add support for different PDF processing modes and strategies
        loader = UnstructuredPDFLoader(
            file_path,
            mode=PDF_PROCESSING_MODE,
            strategy=PDF_PROCESSING_STRATEGY,
            post_processors=[clean_extra_whitespace],
            infer_table_structure=True,
        )

        docs = await loader.aload()
        if not docs or len(docs) == 0:
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
        chunks = text_splitter.split_documents(filtered_docs)
        logger.debug(f"Created {len(chunks)} text chunks from documents.")
    except Exception as e:
        logger.debug(e)
        raise HTTPException(
            status_code=500,
            detail=f"Ha ocurrido un error al obtener la información del archivo: {file_path}",
        )
    return chunks
