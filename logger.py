import json
import logging
from datetime import datetime
from logging import Formatter

import config

logger = logging.getLogger("uvicorn")

log_level = config.app["log_level"]
logger.setLevel(log_level)


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def format(self, record):
        json_record = {
            "level": record.levelname,
            "name": record.name,
            "datetime": datetime.fromtimestamp(record.created).isoformat(),
            "message": record.getMessage(),
        }
        return json.dumps(json_record)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.handlers = [handler]

logger.info(f"Logger initialized with level: {log_level}")
