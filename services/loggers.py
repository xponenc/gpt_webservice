import logging

from services.loggers_config import get_logger

logging.basicConfig(
    level=logging.INFO,
    encoding='utf-8',
    )
logger = get_logger("gpt_bot")
