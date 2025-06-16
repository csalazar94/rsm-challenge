from fastapi import Depends, FastAPI

import dependencies
import loaders
import logger

app = FastAPI()


@app.get("/")
async def root(
    vectorstore=Depends(dependencies.get_postgres_async_vectorstore),
):
    chunks = await loaders.get_chunks_from_pdf("sample.pdf")
    await vectorstore.adelete()
    logger.debug(f"Deleted existing documents in vectorstore.")
    await vectorstore.aadd_documents(chunks)
    logger.debug(f"Added {len(chunks)} documents to vectorstore.")
    return chunks
