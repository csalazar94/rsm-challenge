from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

import config
from database import async_engine

embeddings = OpenAIEmbeddings(model=str(config.openai["embedding_model"]))


def get_postgres_async_vectorstore():
    """
    Create and return an asynchronous PostgreSQL vector store instance.

    Returns:
        PGVector: An async-enabled PGVector instance configured with:
            - Global embeddings model
            - Collection name from config
            - Async database engine connection
            - JSONB support enabled
    """
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=f"{config.vectordb['collection']}",
        connection=async_engine,
        use_jsonb=True,
        async_mode=True,
    )

    return vectorstore
