from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_postgres.vectorstores import PGVector

import config
from logger import logger


async def get_relevant_docs(
    original_question: str, llm: BaseChatModel, vectorstore: PGVector
) -> list[Document]:
    """
    Retrieve relevant documents from a vector database using multi-query retrieval.

    This function enhances document retrieval by:
    1. Translating non-English questions to English for better vector search
    2. Generating multiple variations of the original question to improve retrieval
    3. Using similarity score threshold to filter relevant results

    Args:
        original_question (str): The user's original question in any language
        llm (BaseChatModel): Language model for translation and query generation
        vectorstore (PGVector): PostgreSQL vector database containing documents

    Returns:
        List[Document]: List of relevant documents retrieved from the vector store

    Example:
        >>> docs = await get_relevant_docs(
        ...     "How do I reset my password?",
        ...     llm,
        ...     vectorstore
        ... )
        >>> len(docs)
        5
    """
    logger.info(f"Translating question: {original_question[:100]}...")
    translated_question = await llm.ainvoke(
        f"""If this text is not in English, translate it to English. 
        If it's already in English, return it as is.

        Examples:
        Spanish: "¿Cómo restablezco mi contraseña?" → "How do I reset my password?"
        French: "Comment réinitialiser mon mot de passe?" → "How do I reset my password?"
        English: "How do I reset my password?" → "How do I reset my password?"
        
        Text: {original_question}"""
    )
    logger.debug(f"Translated to: {translated_question.content}")

    logger.debug(
        f"Setting up retriever with score threshold: "
        f"{config.retriever['score_threshold']}"
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": config.retriever["score_threshold"]},
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

    docs = list({doc.metadata["element_id"]: doc for doc in docs}.values())
    logger.info(f"Retrieved {len(docs)} unique documents from vector store")
    logger.debug(
        f"Document IDs: {[doc.metadata.get('element_id', 'N/A') for doc in docs]}"
    )

    return docs
