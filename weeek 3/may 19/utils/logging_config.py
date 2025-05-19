"""
Logging configuration for the Web Research Assistant.
"""
import logging
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_level: The logging level to use.
    """
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"research_assistant_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")
