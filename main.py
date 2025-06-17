import sentry_sdk
from fastapi import Depends, FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import config
import dependencies
import helpers
import loaders
import logger

sentry_sdk.init(
    dsn=config.sentry["dsn"],
    send_default_pii=True,
)

app = FastAPI()


@app.get("/debug_error")
async def debug_error():
    raise Exception("This is a debug error to test Sentry integration.")


@app.get("/health")
async def healthcheck():
    return "OK"


@app.post("/ingest")
async def ingest(
    vectorstore=Depends(dependencies.get_postgres_async_vectorstore),
):
    chunks = await loaders.get_chunks_from_pdf("sample.pdf")
    await vectorstore.adelete()
    logger.debug(f"Deleted existing documents in vectorstore.")
    await vectorstore.aadd_documents(chunks)
    logger.debug(f"Added {len(chunks)} documents to vectorstore.")
    return chunks


class QueryBody(BaseModel):
    question: str = "Chapter 1 is about?"


@app.post("/query")
async def query(
    body: QueryBody,
    vectorstore=Depends(dependencies.get_postgres_async_vectorstore),
):
    llm = ChatOpenAI(
        model=str(config.openai["chat_model"]), temperature=config.llm["temperature"]
    )

    docs = await helpers.get_relevant_docs(body.question, llm, vectorstore)

    if len(docs) == 0:
        return "No relevant information found."

    docs = list({doc.metadata["element_id"]: doc for doc in docs}.values())

    MAX_DOCS = 10
    context = "\n\n".join([doc.page_content for doc in docs[:MAX_DOCS]])

    PROMPT = f"""
    Answer the user question using only the provided information on documents.

    Documents:
    {context}

    Question:
    {body.question}

    Answer in the same language as the original question:
    """

    llm_response = await llm.ainvoke(PROMPT)
    answer = str(llm_response.content)
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
