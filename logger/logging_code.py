"""
This module is a custom logger used for logging actions during main model pipeline run
"""
import logging
from configs.config_business_rules import APP_LOGGER_NAME


class CustomLogger:
    """
    CustomLogger class implements methods that helps other modules to log their actions easily.
    Args:
    _____
        logger_name(str): Name of the app logger
        is_debug(bool): Optional parameter default is False, represents Log level false means INFO.
        file_name(str): File name for the log file
    """
    instances_dict = {}

    def __init__(self, logger_name=APP_LOGGER_NAME, is_debug=False, file_name=None):
        self.logger = logging.getLogger(logger_name)
        logging.basicConfig(level=logging.NOTSET)
        self.logger.setLevel(logging.DEBUG if is_debug else logging.INFO)
        self.file_name = file_name
        if len(self.logger.handlers) == 0:
            self.add_handlers()
        self.instances_dict[file_name] = self

    @staticmethod
    def get_instance(file_name):
        if file_name not in CustomLogger.instances_dict:
            CustomLogger(file_name=file_name)
        return CustomLogger.instances_dict[file_name]

    def log(self, message, level=None):
        """
        Main method to log a message into the logger instance
        Args:
            message(str): message to log
            level(str): log message level default is None which represents INFO or DEBUG base on class initialisation
        Returns:
            None
        """
        if level is None:
            level = self.logger.level
        elif level == 'debug':
            level = logging.DEBUG
        elif level == 'info':
            level = logging.INFO
        elif level == 'warning':
            level = logging.WARNING
        elif level == 'error':
            level = logging.ERROR
        else:
            raise NotImplementedError(f"logging level {level} unsupported")
        self.logger.propagate = False
        self.logger.log(level=level, msg=message)

    def add_handlers(self):
        """
        Method to add handlers into logger instance. If filename is provided then file handler and base handler
        both are added o.w. only console handler is added.
        Returns:
        """
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s -%(module)s - %(funcName)s - %(levelname)s - %(message)s"
        )
        if self.file_name:
            file_handler = logging.FileHandler(self.file_name)
            file_handler.setLevel(level=logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level=logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
