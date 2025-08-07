# Copyright 2025 Hartmut Häntze
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

import logging


class LevelBasedFormatter(logging.Formatter):
    """Custom formatter that uses different formats based on log level"""

    def __init__(self):
        super().__init__()

        # Define different formats for each level
        self.formats = {
            logging.DEBUG: "[DEBUG] %(filename)s:%(lineno)d - %(message)s",
            logging.INFO: "%(asctime)s - %(message)s",
            logging.WARNING: "[WARNING] %(asctime)s %(filename)s:%(lineno)d - %(message)s",
            logging.ERROR: "[ERROR] %(asctime)s %(filename)s:%(lineno)d - %(message)s",
            logging.CRITICAL: "[CRITICAL] %(asctime)s %(filename)s:%(lineno)d - %(message)s",
        }

        # Set date format to match your original
        self.date_format = "%H:%M:%S"

    def format(self, record):
        # Get the format string for this log level
        format_string = self.formats.get(record.levelno, self.formats[logging.INFO])

        # Create a new formatter with the appropriate format
        formatter = logging.Formatter(format_string, self.date_format)

        return formatter.format(record)


def setup_logging(level=logging.INFO):
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear any existing handlers (important to avoid duplicates)
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Apply custom formatter
    formatter = LevelBasedFormatter()
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)


# Test the setup
if __name__ == "__main__":
    setup_logging()

    logger = logging.getLogger(__name__)

    logger.debug("This is a debug message")  # Won't show because level is INFO
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
