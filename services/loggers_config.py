import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


class LevelFileHandler(logging.Handler):
    """Логирование разных уровней сообщений в разные файлы"""
    LEVELS = {
        50: "CRITICAL",
        40: "ERROR",
        30: "WARNING",
        20: "INFO",
        10: "DEBUG",
        0: "NOTSET",
    }

    def __init__(self, file_name="calc_"):
        self.file_name = file_name
        super().__init__()

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)

        with open(self.file_name + self.LEVELS.get(record.levelno, "") + ".log", mode="a", encoding="utf-8") as f:
            f.write(message + '\n')


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    if logger_name == 'gpt_bot':
        # level_handler = LevelFileHandler(file_name="gpt_bot_")
        logs_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        if not os.path.exists(logs_directory):
            os.makedirs(logs_directory)
        level_handler = TimedRotatingFileHandler(os.path.join(logs_directory, "gpt_bot.log"), when='D', interval=1,
                                                 backupCount=10, encoding='utf-8', )
        stream_format = logging.Formatter(fmt="%(levelname)s | %(name)s | %(asctime)s | %(filename)s |"
                                              " %(lineno)s | %(message)s")
        level_handler.setFormatter(stream_format)
        logger.addHandler(level_handler)
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setFormatter(stream_format)
        logger.addHandler(stream)
        logger.propagate = False
        return logger

    elif logger_name == 'utils':
        level_handler = LevelFileHandler()
        stream_format = logging.Formatter(fmt="%(levelname)s | %(name)s | %(asctime)s | %(filename)s |"
                                              " %(lineno)s | %(message)s")
        level_handler.setFormatter(stream_format)
        logger.addHandler(level_handler)
        logger.propagate = False

