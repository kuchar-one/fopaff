import logging
import sys
import os


def setup_logger():
    """
    Set up logging for the quantum operations library.

    The logger is set up as follows:
    - All log messages are written to a file named "quantum_operations.log" in the current directory.
    - Log messages with level WARNING and higher are shown in the terminal.
    - Log messages with level INFO and lower are not shown in the terminal.

    Returns:
        The configured logger object.
    """
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logger.setLevel(logging.DEBUG)

    # File handler for all logs
    file_handler = logging.FileHandler("output/quantum_operations.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Stream handler for errors and warnings
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    stream_handler.setLevel(
        logging.INFO
    )  # Only show warnings and errors in terminal
    logger.addHandler(stream_handler)

    return logger
