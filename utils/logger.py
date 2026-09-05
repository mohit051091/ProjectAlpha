"""
Logging configuration
"""

import logging
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_dir: str = "./logs", level=logging.INFO) -> logging.Logger:
    """
    Setup a logger with file and console handlers
    
    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    log_dir : str
        Directory to store log files
    level : int
        Logging level (logging.INFO, logging.DEBUG, etc.)
    
    Returns
    -------
    logging.Logger
        Configured logger
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler (detailed)
    log_file = Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simple)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


# Module-level logger
logger = setup_logger(__name__)
