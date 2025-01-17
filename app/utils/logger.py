import logging
import os

def get_logger(name: str, log_level=logging.DEBUG) -> logging.Logger:
    """
    Creates and configures a logger with a specified name and log level.
    
    Args:
        name (str): Name of the logger.
        log_level (int): Logging level (default: logging.DEBUG).
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Check if the logger already has handlers (to prevent duplicate logs)
    if not logger.handlers:
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create a file handler (logs will be written to a file)
        log_directory = "logs"
        os.makedirs(log_directory, exist_ok=True)
        log_file_path = os.path.join(log_directory, f"{name}.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)

        # Create a formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
