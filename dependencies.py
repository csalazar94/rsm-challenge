from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

import config
from database import async_engine
from logger import logger

embeddings = OpenAIEmbeddings(model=str(config.openai["embedding_model"]))


def get_postgres_async_vectorstore():
    """
    Create and return an asynchronous PostgreSQL vector store instance.
    """
    try:
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=f"{config.vectordb['collection']}",
            connection=async_engine,
            use_jsonb=True,
            async_mode=True,
        )
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to create vectorstore: {str(e)}")
        raise HTTPException(status_code=500, detail="Database connection failed")
