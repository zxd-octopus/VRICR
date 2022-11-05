
"""
usage:
"""

import logging


def get_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s [%(filename)s %(lineno)d] %(name)s - %(levelname)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    # two handlers
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(handler)
    return logger


if __name__ == '__main__':
    logger = get_logger("main", "../dump/log/log.txt")
    logger.info("hello")
    logger.debug("word")

    logger_sub = logging.getLogger("main.sub")
    logger_sub.info("hello sub")
    logger_sub.debug("word sub")