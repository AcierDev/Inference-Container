# config.py
import os
from pathlib import Path
from typing import Set, Dict, Any
import logging
from dataclasses import dataclass

# Maintain original direct variables for backward compatibility
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB default

@dataclass
class FileConfig:
    """File upload configuration settings"""
    upload_folder: Path
    allowed_extensions: Set[str]
    max_content_length: int
    temp_file_timeout: int = 3600

@dataclass
class SecurityConfig:
    """Security-related configuration settings"""
    api_key_header: str = 'X-API-Key'
    min_api_key_length: int = 32
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600
    trusted_proxies: Set[str] = frozenset({'127.0.0.1'})

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    log_folder: Path
    max_log_size: int = 10 * 1024 * 1024  # 10MB default
    backup_count: int = 5
    log_level: str = 'INFO'

class Config:
    """Application configuration with environment variable support"""
    
    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent.resolve()
        
        # File handling configuration
        self.file = FileConfig(
            upload_folder=Path(os.getenv('UPLOAD_FOLDER', UPLOAD_FOLDER)),
            allowed_extensions=set(os.getenv('ALLOWED_EXTENSIONS', ','.join(ALLOWED_EXTENSIONS)).split(',')),
            max_content_length=int(os.getenv('MAX_CONTENT_LENGTH', MAX_CONTENT_LENGTH)),
            temp_file_timeout=int(os.getenv('TEMP_FILE_TIMEOUT', 3600))
        )
        
        # Initialize other configs...
        self.security = SecurityConfig()
        self.logging = LoggingConfig(
            log_folder=Path(os.getenv('LOG_FOLDER', self.BASE_DIR / 'logs'))
        )
        
        # Server configuration
        self.SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
        self.SERVER_PORT = int(os.getenv('SERVER_PORT', 5000))
        
        # Initialize directories
        self._init_directories()
    
    def _init_directories(self) -> None:
        """Initialize required directories"""
        try:
            self.file.upload_folder.mkdir(parents=True, exist_ok=True)
            self.logging.log_folder.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to initialize directories: {str(e)}")
            raise

# Create global instance
config = Config()