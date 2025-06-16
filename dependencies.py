from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

import config
from database import async_engine

embeddings = OpenAIEmbeddings(model=str(config.openai["embedding_model"]))


def get_postgres_async_vectorstore():
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=f"{config.vectordb['collection']}",
        connection=async_engine,
        use_jsonb=True,
        async_mode=True,
    )

    return vectorstore
