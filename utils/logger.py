import logging
import os
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler
from config.config import LOGGING_CONFIG
from colorama import Fore, Style

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def delete_existing_logs():
    """
    Mevcut log dosyalarını siler
    """
    try:
        if os.path.exists('logs'):
            for file in os.listdir('logs'):
                if file.endswith('.log'):
                    file_path = os.path.join('logs', file)
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Log dosyası silinirken hata: {str(e)}")
    except Exception as e:
        print(f"Log dosyaları silinirken hata: {str(e)}")

def setup_logger():
    """
    Configures and returns the main logger.
    Uses LOGGING_CONFIG from config/config.py.
    """
    # Log dosyalarının otomatik silinmesi kaldırıldı
    # delete_existing_logs()
    
    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Configure root logger
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add file handler
    log_file = f'logs/trading_bot_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Add error file handler
    error_log_file = 'logs/error.log'
    error_file_handler = RotatingFileHandler(
        error_log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(file_formatter)
    logger.addHandler(error_file_handler)
    
    return logger

# Create main logger instance
logger = setup_logger()

def print_section(message: str, log_message: Optional[str] = None):
    """Prints a section header"""
    divider = "=" * 60
    centered_message = message.center(60)
    formatted_section = f"\n{divider}\n{centered_message}\n{divider}"
    logger.info(log_message if log_message else message)
    print(Fore.CYAN + formatted_section + Style.RESET_ALL)

def print_info(message: str, log_message: Optional[str] = None, additional_info: Optional[str] = None, log_additional: Optional[str] = None) -> None:
    """Log an info message with optional additional information."""
    formatted_message = f"{Colors.BLUE}ℹ️  {message.ljust(60)}{Colors.ENDC}"
    print(formatted_message)
    if additional_info:
        print(f"{Colors.BLUE}   {additional_info.ljust(60)}{Colors.ENDC}")
    logger.info(log_message if log_message else message)
    if log_additional:
        logger.info(log_additional)
    elif additional_info:
        logger.info(additional_info)

def print_warning(message: str, log_message: Optional[str] = None, additional_info: Optional[str] = None, log_additional: Optional[str] = None) -> None:
    """Log a warning message with optional additional information."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")
    if additional_info:
        print(f"{Colors.YELLOW}   {additional_info}{Colors.ENDC}")
    logger.warning(log_message if log_message else message)
    if log_additional:
        logger.warning(log_additional)
    elif additional_info:
        logger.warning(additional_info)

def print_error(message: str, log_message: Optional[str] = None, additional_info: Optional[str] = None, log_additional: Optional[str] = None) -> None:
    """Log an error message with optional additional information."""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")
    if additional_info:
        print(f"{Colors.RED}   {additional_info}{Colors.ENDC}")
    logger.error(log_message if log_message else message)
    if log_additional:
        logger.error(log_additional)
    elif additional_info:
        logger.error(additional_info)

def print_success(message: str, log_message: Optional[str] = None, additional_info: Optional[str] = None, log_additional: Optional[str] = None) -> None:
    """Log a success message with optional additional information."""
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")
    if additional_info:
        print(f"{Colors.GREEN}   {additional_info}{Colors.ENDC}")
    logger.info(log_message if log_message else message)
    if log_additional:
        logger.info(log_additional)
    elif additional_info:
        logger.info(additional_info)

def print_balance_info(message: str) -> None:
    """Print balance related information with special formatting."""
    formatted_message = f"{Colors.CYAN}💰 {message.ljust(60)}{Colors.ENDC}"
    print(formatted_message)
    logger.info(message)

def print_trade_info(message: str) -> None:
    """Print trade related information with special formatting."""
    formatted_message = f"{Colors.BOLD}{Colors.GREEN}📊 {message.ljust(60)}{Colors.ENDC}"
    print(formatted_message)
    logger.info(message)

def setup_epoch_logger():
    """
    Model eğitimi sırasında her epoch'un sonuçlarını loglamak için özel bir logger oluşturur.
    """
    # Create logs directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Epoch logger oluştur
    epoch_logger = logging.getLogger('TradingBot.EpochTracker')
    epoch_logger.setLevel(logging.INFO)
    
    # Epoch log dosyası
    epoch_log_path = os.path.join('logs', f'training_epochs.log')
    epoch_file_handler = RotatingFileHandler(
        epoch_log_path, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    # Log formatı
    epoch_formatter = logging.Formatter('%(asctime)s - %(message)s')
    epoch_file_handler.setFormatter(epoch_formatter)
    
    # Önceki handler'ları temizle
    if epoch_logger.handlers:
        epoch_logger.handlers.clear()
    
    epoch_logger.addHandler(epoch_file_handler)
    
    return epoch_logger

# Epoch logger'ı oluştur
epoch_logger = setup_epoch_logger()

def log_epoch_results(epoch, epochs, train_loss, train_acc, val_loss=None, val_acc=None, lr=None):
    """
    Her epoch sonunda sonuçları loglama fonksiyonu
    """
    # Temel epoch bilgileri
    log_message = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2%}"
    
    # Validation bilgileri varsa ekle
    if val_loss is not None and val_acc is not None:
        log_message += f", Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2%}"
    
    # Learning rate bilgisi varsa ekle
    if lr is not None:
        log_message += f", LR: {lr:.2e}"
    
    # Epoch logger'a kaydet
    epoch_logger.info(log_message) 