import logging
import sys
import os
from logging import Logger


def setup_logger() -> Logger:
    """
    Configure and set up logging for the quantum operations library.

    This function creates a logger with two handlers:
    1. File Handler:
       - Writes all logs (INFO and above) to 'output/quantum_operations.log'
       - Uses detailed formatting including timestamp, filename, and level
       - Creates the output directory if it doesn't exist
       - Appends to existing log file if present

    2. Stream Handler:
       - Writes WARNING and above to stdout (terminal)
       - Uses simplified formatting for better readability
       - Filters out INFO and DEBUG messages from terminal output

    The log format for file output is:
    "%(asctime)s - %(filename)s - %(levelname)s - %(message)s"

    The log format for terminal output is:
    "%(levelname)s - %(message)s"

    Returns:
        logging.Logger: Configured logger instance ready for use.

    Example:
        >>> logger = setup_logger()
        >>> logger.info("Starting quantum operation")  # Goes to file only
        >>> logger.warning("Memory usage high")        # Goes to both file and terminal
        >>> logger.error("Operation failed")           # Goes to both file and terminal

    Note:
        - The logger name is set to the basename of the executing script
        - Log files are stored in the 'output' directory
        - Previous log files are preserved (logs are appended)
        - Terminal output is color-coded by level (if supported)
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Get logger instance with script name
    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logger.setLevel(logging.INFO)

    # File handler for all logs
    file_handler = logging.FileHandler("output/quantum_operations.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Stream handler for errors and warnings
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
    stream_handler.setLevel(
        logging.WARNING
    )  # Only show warnings and errors in terminal
    logger.addHandler(stream_handler)

    return logger
