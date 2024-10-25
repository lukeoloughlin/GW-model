import os
import sys
import logging
import argparse


# If extra={"simple": True} is passed to logger.info, then the message logged is just the string passed
class ConditionalFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, "simple") and record.simple:
            return record.getMessage()
        else:
            return logging.Formatter.format(self, record)


def create_log(fname: str, path: str | None = None) -> logging.Logger:
    if path is not None:
        fname = os.path.join(path, fname)

    # create the log file
    if not os.path.isfile(fname):
        open(fname, "w+").close()

    logging_format = "%(levelname)s: %(asctime)s: %(message)s"

    # configure loger
    logging.basicConfig(filename=fname, level=logging.INFO, format=logging_format)
    logger = logging.getLogger()

    formatter = ConditionalFormatter(logging_format)
    logger.handlers[0].setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(stream_handler)

    # create file handler
    # handler = logging.FileHandler(fname)

    # set logging level
    # handler.setLevel(logging.INFO)

    # create formatter for logger
    # logger.addHandler(handler)

    return logger


def log_args(
    logger: logging.Logger, args: argparse.Namespace, indent: bool = False
) -> None:
    if indent:
        prefix = "\t"
    else:
        prefix = ""
    logger.info("Arguments:", extra={"simple": True})
    for argname, argval in vars(args).items():
        logger.info(prefix + f"{argname}: {argval}", extra={"simple": True})
    logger.info("", extra={"simple": True})  # Line break


def log_dict(logger: logging.Logger, kv_pairs: dict, indent: bool = False) -> None:
    if indent:
        prefix = "\t"
    else:
        prefix = ""
    for k, v in kv_pairs.items():
        logger.info(prefix + f"{k}: {v}", extra={"simple": True})
    logger.info("", extra={"simple": True})  # Line break
