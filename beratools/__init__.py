from beratools.core import *
from beratools.tools import *
from beratools.gui import *
from beratools.third_party import *

import logging
import logging.handlers
import sys


def print(msg, flush=True):
    log = logging.getLogger()
    log.info(msg)

    if flush:
        for handler in log.handlers:
            handler.flush()


class NoParsingFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("parsing")


def setup_logger(name, log_file):
    # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)

    # Add stdout handler, with level INFO
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)-8s %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    # Add file rotating handler, with level DEBUG
    rotating_handler = logging.handlers.RotatingFileHandler(
        filename=log_file, maxBytes=2*1000*1000, backupCount=5
    )

    rotating_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    rotating_handler.setFormatter(formatter)
    logging.getLogger().addHandler(rotating_handler)

    logging.getLogger().addFilter(NoParsingFilter())
    return logging.getLogger()


# log = setup_logger('', r'D:\Temp\logging\rotation.log')
# log.debug("Debug message, should only appear in the file.")