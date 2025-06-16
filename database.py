from sqlalchemy.ext.asyncio import create_async_engine

import config

SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg://{config.db['user']}:{config.db['password']}@{config.db['host']}:{config.db['port']}/{config.db['name']}"
async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=config.db["pool_size"],
    pool_pre_ping=config.db["pool_pre_ping"],
    pool_recycle=config.db["pool_recycle"],
)
