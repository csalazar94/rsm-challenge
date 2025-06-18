import os

import sentry_sdk
from fastapi import Depends, FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
import dependencies
import helpers
import loaders
from logger import logger

if config.sentry["dsn"]:
    sentry_sdk.init(
        dsn=config.sentry["dsn"],
        send_default_pii=True,
    )


app = FastAPI()


@app.get("/debug_error")
async def debug_error():
    logger.error("Debug error endpoint triggered")
    raise Exception("This is a debug error to test Sentry integration.")


@app.get("/health")
async def healthcheck():
    logger.info("Health check requested")
    return "OK"


@app.post("/ingest")
async def ingest(
    vectorstore=Depends(dependencies.get_postgres_async_vectorstore),
):
    logger.info("Starting document ingestion process")
    try:
        await vectorstore.adelete()
        logger.debug(f"Deleted existing documents in vectorstore.")

        file_names = os.listdir("files")
        for file_name in file_names:
            file_path = os.path.join("files", file_name)
            chunks = await loaders.get_chunks_from_file(file_path)
            logger.info(f"Loaded {len(chunks)} chunks from file: {file_path}")
            await vectorstore.aadd_documents(chunks)
            logger.debug(f"Added {len(chunks)} documents to vectorstore.")
            logger.info("Document ingestion completed successfully")
    except Exception as e:
        logger.error(f"Document ingestion failed: {str(e)}")
        raise e


class QueryBody(BaseModel):
    question: str = Field(
        min_length=1, max_length=1000, description="Question to ask about the documents"
    )


@app.post("/query")
async def query(
    body: QueryBody,
    vectorstore=Depends(dependencies.get_postgres_async_vectorstore),
):
    logger.info(f"Query received: {body.question[:50]}...")
    try:
        llm = ChatOpenAI(
            model=str(config.openai["chat_model"]),
            temperature=config.llm["temperature"],
        )

        docs = await helpers.get_relevant_docs(body.question, llm, vectorstore)
        logger.debug(f"Retrieved {len(docs)} relevant documents")

        if len(docs) == 0:
            logger.warning("No relevant documents found for query")
            return "No relevant information found."

        MAX_DOCS = 10
        context = "\n\n".join([doc.page_content for doc in docs[:MAX_DOCS]])

        PROMPT = f"""
        Answer the user question using only the provided information on documents.

        Documents:
        {context}

        Question:
        {body.question}

        Answer:
        """

        llm_response = await llm.ainvoke(PROMPT)
        answer = str(llm_response.content)
        logger.info(f"Generated answer with {len(docs)} sources")
        sources = [
            {
                "filename": doc.metadata.get(
                    "source", None
                ),  # not explicitly on instructions, but added for clarity
                "page": doc.metadata.get("page_number", None),
                "text": doc.page_content,
            }
            for doc in docs
        ]

        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise e
