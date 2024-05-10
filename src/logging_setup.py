import logging

def setup_logging() -> None:
    """
    Sets up logger
    
    Args:
        None

    Returns:
        None
    """
    logger = logging.getLogger()
    if not logger.handlers:  # Prevent adding handlers multiple times
        logger.setLevel(logging.INFO)

        # Formatter for our handlers
        formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(filename)s:%(lineno)d  - %(levelname)s - %(message)s', 
                            datefmt='%Y-%m-%d %H:%M:%S')

        # File handler
        file_handler = logging.FileHandler('app.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream (console) handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

