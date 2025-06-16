from fastapi import Depends, FastAPI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

import config
import dependencies
import loaders
import logger

app = FastAPI()


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

    translated_question = await llm.ainvoke(
        f"Translate this question to English: {body.question}"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1}
    )
    prompt_retriever = PromptTemplate.from_template(
        """
        OBJECTIVE: Generate 1-5 variations of the original question to 
        improve retrieval of relevant documents from vector database.

        STRATEGY: Overcome similarity search limitations through query 
        diversification.

        GENERATION INSTRUCTIONS:
        1. BREAKDOWN: If the question contains multiple concepts, create 
           a sub-question for each concept
        2. REFORMULATION: Generate alternative versions with synonyms or 
           different structures
        3. SPECIFICATION: Create more specific or general variants based 
           on context

        QUALITY CRITERIA:
        - Maintain original search intent
        - Ensure each variant provides unique value for retrieval

        REQUIRED FORMAT:
        One question per line, no numbering or additional formatting.

        EXAMPLES:
        Original question: "How do I reset my password and enable 2FA?"
        Generated variations:
        How can I reset my account password?
        What steps are needed to enable two-factor authentication?
        How do I change my login credentials?
        What is the process for account security setup?
        How do I recover access to my account?

        ORIGINAL QUESTION: {question}
        """
    )
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=prompt_retriever,
        include_original=True,
    )

    docs = await retriever_from_llm.ainvoke(str(translated_question.content))

    if len(docs) == 0:
        return "No relevant information found."

    docs = list({doc.metadata["element_id"]: doc for doc in docs}.values())

    MAX_DOCS = 10
    context = "\n\n".join([doc.page_content for doc in docs[:MAX_DOCS]])

    PROMPT = f"""
    Answer the question based only on the context provided in the documents.

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
