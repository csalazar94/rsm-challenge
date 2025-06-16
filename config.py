import os

from dotenv import load_dotenv

load_dotenv()

app = dict(
    environment=os.getenv("APP_ENVIRONMENT", "development"),
)

db = dict(
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "password"),
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    name=os.getenv("DB_NAME", "postgres"),
    pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
)

vectordb = dict(
    collection=os.getenv("VECTORDB_COLLECTION", "documents"),
)

openai = dict(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
    chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
    chat_model_date=os.getenv("OPENAI_CHAT_MODEL_DATE", "2024-05-31"),
)

llm = dict(
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
)

sentry = dict(
    dsn=os.getenv("SENTRY_DSN", ""),
)

cors = dict(
    allowed_origins=os.getenv("CORS_ALLOWED_ORIGINS", "").split(","),
)
