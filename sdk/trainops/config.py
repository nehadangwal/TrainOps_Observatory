"""
Configuration and logging setup for TrainOps SDK
"""
import os
import logging
from typing import Optional


class Config:
    """TrainOps configuration"""
    
    # API Configuration
    API_URL = os.getenv('TRAINOPS_API_URL', 'http://localhost:5000')
    API_TIMEOUT = int(os.getenv('TRAINOPS_API_TIMEOUT', '10'))
    
    # Collection intervals
    COLLECT_INTERVAL = int(os.getenv('TRAINOPS_COLLECT_INTERVAL', '10'))  # seconds
    SEND_INTERVAL = int(os.getenv('TRAINOPS_SEND_INTERVAL', '30'))  # seconds
    BUFFER_SIZE = int(os.getenv('TRAINOPS_BUFFER_SIZE', '500'))
    
    # Retry configuration
    MAX_RETRIES = int(os.getenv('TRAINOPS_MAX_RETRIES', '3'))
    RETRY_DELAY = int(os.getenv('TRAINOPS_RETRY_DELAY', '5'))  # seconds
    
    # Logging
    LOG_LEVEL = os.getenv('TRAINOPS_LOG_LEVEL', 'INFO')
    
    # Feature flags
    ENABLE_GPU_METRICS = os.getenv('TRAINOPS_ENABLE_GPU', 'true').lower() == 'true'
    FAIL_ON_API_ERROR = os.getenv('TRAINOPS_FAIL_ON_ERROR', 'false').lower() == 'true'


def setup_logging(level: Optional[str] = None):
    """Setup logging configuration"""
    log_level = level or Config.LOG_LEVEL
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set TrainOps logger
    logger = logging.getLogger('trainops')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


# Initialize logger
logger = setup_logging()
