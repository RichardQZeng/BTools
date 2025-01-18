import logging
import logging.handlers
import sys
from beratools.gui.bt_data import BTData

bt = BTData()


class NoParsingFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("parsing")


class Logger(object):
    def __init__(self, name, console_level=logging.INFO, file_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.name = name
        self.console_level = console_level
        self.file_level = file_level

        self.setup_logger()

    def get_logger(self):
        return self.logger

    def print(self, msg, flush=True):
        """
        This is for including print in logging
        Parameters
        ----------
        msg :
        flush :

        Returns
        -------

        """
        self.logger.info(msg)
        if flush:
            for handler in self.logger.handlers:
                handler.flush()

    def setup_logger(self):
        """
        # log = setup_logger('', r'PATH_TO_LOG_FILE')
        # log.debug("Debug message, should only appear in the file.")

        # for i in range(0, 10000):
        #     print("From print(): Info message, should appear in file and stdout.")
        #     log.info("Info message, should appear in file and stdout.")
        #     log.warning("Warning message, should appear in file and stdout.")
        #     log.error("Error message, should appear in file and stdout.")
        #     log.error("parsing, should appear in file and stdout.")
        Parameters
        ----------
        name :
        console_level :
        file_level :

        Returns
        -------

        """
        # Change root logger level from WARNING (default) to NOTSET in order for all messages to be delegated.
        logging.getLogger().setLevel(logging.NOTSET)
        log_file = bt.get_logger_file_name(self.name)

        # Add stdout handler, with level INFO
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

        # Add file rotating handler, 5MB size limit, 5 backups
        rotating_handler = logging.handlers.RotatingFileHandler(
            filename=log_file, maxBytes=5*1000*1000, backupCount=5
        )

        rotating_handler.setLevel(self.file_level)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        rotating_handler.setFormatter(formatter)
        logging.getLogger().addHandler(rotating_handler)
        logging.getLogger().addFilter(NoParsingFilter())
