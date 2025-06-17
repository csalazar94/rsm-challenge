import logging

import config

logger = logging.getLogger("uvicorn")
log_level = config.app["log_level"]
logger.setLevel(log_level)
logger.info(f"Logger initialized with level: {log_level}")
