import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging(log_filename: str, log_dir: str = '.'):
    """
    Configures a rotating file logger for the application.

    This function sets up a global logger that writes to a specified file with
    log rotation. It is designed to be idempotent, preventing the addition of
    duplicate handlers if called multiple times, which is crucial for environments
    like Streamlit that re-run scripts.

    Args:
        log_filename: The name of the log file (e.g., 'app.log').
        log_dir: The directory to store the log file. Defaults to the current directory.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times by checking if a handler for the same file already exists.
    if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == log_filepath for h in root_logger.handlers):
        max_bytes = 10 * 1024 * 1024  # 10 MB
        backup_count = 5  # Keep 5 backup log files
        handler = RotatingFileHandler(log_filepath, maxBytes=max_bytes, backupCount=backup_count)

        formatter = logging.Formatter("[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)