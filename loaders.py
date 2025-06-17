import os

from fastapi import HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredRSTLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from unstructured.cleaners.core import clean_extra_whitespace

from logger import logger

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 0
PDF_PROCESSING_MODE = "elements"
PDF_PROCESSING_STRATEGY = "hi_res"


async def get_chunks_from_file(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Asynchronously process a file and split it into document chunks.

    This function loads a supported file (PDF or RST), processes it using the appropriate
    loader, filters metadata, and splits the content into chunks for downstream processing
    such as vector embeddings or search indexing.

    Args:
        file_path (str): Path to the file to be processed. Must be an existing file
            with a supported extension (.pdf or .rst).
        chunk_size (int, optional): Maximum size of each text chunk in characters.
            Defaults to DEFAULT_CHUNK_SIZE.
        chunk_overlap (int, optional): Number of characters to overlap between
            consecutive chunks to maintain context. Defaults to DEFAULT_CHUNK_OVERLAP.

    Returns:
        list[Document]: List of Document objects representing the chunked content.
            Each Document contains the text content and associated metadata.

    Raises:
        HTTPException:
            - 404 if the file does not exist
            - 400 if the file type is unsupported or no documents found
            - 413 if the file exceeds the 50MB size limit
            - 500 if an unexpected error occurs during processing

    Note:
        Supported file types: PDF (.pdf) and reStructuredText (.rst)
        Maximum file size: 50MB

    Example:
        >>> chunks = await get_chunks_from_file("document.pdf", chunk_size=1000, chunk_overlap=200)
        >>> print(f"Generated {len(chunks)} chunks")
    """
    try:
        _, ext = os.path.splitext(file_path)

        logger.info(f"Starting {ext} processing for: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {file_path}",
            )

        supported_extensions = [".pdf", ".rst"]
        if ext.lower() not in supported_extensions:
            logger.error(f"Unsupported file type: {file_path}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_path}. Only {supported_extensions} files are supported.",
            )

        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the limit of 50MB: {file_path}",
            )

        match ext.lower():
            case ".pdf":
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
            case ".rst":
                loader = UnstructuredRSTLoader(
                    file_path, post_processors=[clean_extra_whitespace]
                )
                logger.debug("Configured RST loader with clean_extra_whitespace.")
            case _:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_path}. Only PDF files are supported.",
                )

        docs = await loader.aload()
        if not docs or len(docs) == 0:
            logger.warning(f"No documents found in file: {file_path}")
            raise HTTPException(
                status_code=400,
                detail=f"No documents found in the file: {file_path}",
            )
        logger.debug(f"Loaded {len(docs)} documents from file.")

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
            f"Successfully processed file: {file_path} -> " f"{len(chunks)} chunks"
        )
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing file: {file_path}",
        )
    return chunks
