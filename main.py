from fastapi import FastAPI

import loaders

app = FastAPI()


@app.get("/")
async def root():
    return await loaders.get_chunks_from_pdf("sample.pdf")
