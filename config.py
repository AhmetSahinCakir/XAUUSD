import os
from dotenv import load_dotenv
import json
from pathlib import Path
import logging

# Logger oluştur
logger = logging.getLogger("TradingBot.Config")

# .env dosyasını yükle
load_dotenv()

# MT5 yapılandırması - hassas bilgiler environment variables'dan alınıyor
MT5_CONFIG = {
    'login': int(os.getenv('MT5_LOGIN')),
    'password': os.getenv('MT5_PASSWORD'),
    'server': os.getenv('MT5_SERVER', 'MetaQuotes-Demo')
}

# İşlem parametreleri
TRADING_CONFIG = {
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '1.0')),    # İşlem başına risk yüzdesi (1%)
    'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '5.0')),    # Maksimum günlük zarar yüzdesi (5%)
    'max_total_loss': float(os.getenv('MAX_TOTAL_LOSS', '15.0')),   # Maksimum toplam zarar yüzdesi (15%)
    'transaction_fee': float(os.getenv('TRANSACTION_FEE', '0.00025')), # İşlem ücreti (spread dahil ~2.5 pip)
    'trailing_stop': True,  # Trailing stop kullan
    'trailing_stop_factor': 1.5,  # ATR'nin kaç katı trailing stop mesafesi
    'partial_tp_enabled': True,  # Kısmi kar alma aktif
    'partial_tp_levels': [  # Kısmi kar alma seviyeleri
        {'percentage': 33, 'at_price_factor': 1.5},  # Pozisyonun %33'ü 1.5R'de
        {'percentage': 33, 'at_price_factor': 2.0},  # Diğer %33'ü 2R'de
        {'percentage': 34, 'at_price_factor': 3.0}   # Kalan %34'ü 3R'de
    ]
}

# Model parametreleri
MODEL_CONFIG = {
    'lstm': {
        'input_size': 32,  # Tüm özellikler
        'hidden_size': 256,  # Orta seviye karmaşıklık
        'num_layers': 3,    # 3 katman
        'dropout': 0.3,     # Dropout oranı
        'bidirectional': True,  # Çift yönlü LSTM
        'threshold_high': 0.65,  # Alım sinyali için eşik
        'threshold_low': 0.35,   # Satım sinyali için eşik
        'gradient_clip': 1.0,    # Gradient clipping değeri
    },
    'attention': {
        'dims': [512, 64, 1],  # Attention boyutları düzeltildi
        'dropout': 0.2  # Attention için dropout
    },
    'batch_norm': {
        'momentum': 0.1,
        'eps': 1e-5
    },
    'training': {
        'batch_size': 64,        # Batch size
        'epochs': 100,           # Epoch sayısı
        'learning_rate': 0.001,  # Öğrenme oranı
        'sequence_length': 60,   # Zaman serisi uzunluğu
        'prediction_steps': 1,   # Tahmin adımı
        'train_split': 0.8,      # Eğitim seti oranı
        'validation_split': 0.1, # Validasyon seti oranı
        'test_split': 0.1,      # Test seti oranı
        'early_stopping_patience': 15,  # Early stopping sabır sayısı
        'reduce_lr_patience': 8,        # Learning rate azaltma sabır sayısı
        'reduce_lr_factor': 0.5,        # Learning rate azaltma faktörü
        'min_lr': 1e-6                  # Minimum learning rate
    }
}

# Veri parametreleri
DATA_CONFIG = {
    'timeframes': ['5m', '15m', '1h'],  # Daha az gürültülü zaman dilimleri
    'default_candles': {
        '5m': 1000,
        '15m': 800,
        '1h': 500
    },
    'training_candles': {  # Optimize edilmiş eğitim veri miktarları
        '5m': 1000,
        '15m': 800,
        '1h': 500
    },
    'retraining_interval_days': 7,
    'min_required_candles': 30,
    'feature_columns': [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'ema_fast', 'ema_slow', 'atr', 'bbands_upper',
        'bbands_middle', 'bbands_lower', 'stoch_k',
        'stoch_d', 'adx', 'cci', 'mfi', 'obv',
        'gap', 'gap_size',
        'session_asia', 'session_europe', 'session_us',
        'sentiment_1', 'sentiment_2'
    ]
}

# Sistem parametreleri
SYSTEM_CONFIG = {
    'gc_interval': 300,  # 5 dakikada bir garbage collection
    'max_memory_usage': 0.85,  # Maksimum bellek kullanım oranı
    'reconnect_attempts': 3,  # MT5 yeniden bağlanma deneme sayısı
    'reconnect_wait': 5,  # Yeniden bağlanma bekleme süresi (saniye)
    'heartbeat_interval': 60,  # Bağlantı kontrol aralığı (saniye)
    'emergency_close_timeout': 300,  # Acil kapatma zaman aşımı (saniye)
}

# Loglama yapılandırması
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'filters': {
        'sensitive_data': {
            '()': 'utils.log_filters.SensitiveDataFilter'
        }
    },
    'handlers': {
        'console': {
            'level': 'WARNING',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'filters': ['sensitive_data']
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/trading_bot.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'filters': ['sensitive_data']
        },
        'error_file': {
            'level': 'ERROR',
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/error.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'filters': ['sensitive_data']
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO'
        },
        'TradingBot': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# Piyasa saatleri yapılandırması
MARKET_HOURS = {
    'regular': {
        'open': {'day': 'SUNDAY', 'time': '22:00', 'timezone': 'GMT'},
        'close': {'day': 'FRIDAY', 'time': '22:00', 'timezone': 'GMT'}
    },
    'sessions': {
        'sydney': {
            'open': {'time': '22:00', 'timezone': 'GMT'},  # GMT+10
            'close': {'time': '07:00', 'timezone': 'GMT'}
        },
        'tokyo': {
            'open': {'time': '00:00', 'timezone': 'GMT'},  # GMT+9
            'close': {'time': '09:00', 'timezone': 'GMT'}
        },
        'london': {
            'open': {'time': '08:00', 'timezone': 'GMT'},
            'close': {'time': '17:00', 'timezone': 'GMT'}
        },
        'new_york': {
            'open': {'time': '13:00', 'timezone': 'GMT'},  # GMT-5
            'close': {'time': '22:00', 'timezone': 'GMT'}
        }
    },
    'early_close': {
        'US': {'time': '13:00', 'timezone': 'EST'}
    },
    'trading_breaks': {
        'daily': [
            # Günlük bakım/rollover periyodu
            {'start': {'time': '21:55', 'timezone': 'GMT'},
             'end': {'time': '22:05', 'timezone': 'GMT'},
             'description': 'Daily maintenance/rollover period'}
        ]
    },
    'weekend_breaks': {
        'start': {'day': 'FRIDAY', 'time': '22:00', 'timezone': 'GMT'},
        'end': {'day': 'SUNDAY', 'time': '22:00', 'timezone': 'GMT'}
    }
}

# Piyasa durumu kontrol aralıkları
MARKET_CHECK_INTERVALS = {
    'session_check': 300,  # 5 dakikada bir seans kontrolü
    'holiday_check': 3600,  # Saatte bir tatil günü kontrolü
    'maintenance_check': 60  # Her dakika bakım/rollover kontrolü
}

def load_market_holidays():
    """Piyasa tatil günlerini yükle"""
    holiday_file = Path('data/market_holidays.json')
    if not holiday_file.exists():
        logger.warning("market_holidays.json bulunamadı, varsayılan tatil günleri kullanılacak")
        MARKET_HOURS['holidays'] = []
    else:
        try:
            with open(holiday_file, 'r') as f:
                MARKET_HOURS['holidays'] = json.load(f)
                logger.info(f"{len(MARKET_HOURS['holidays'])} tatil günü yüklendi")
        except Exception as e:
            logger.error(f"Tatil günleri yüklenirken hata: {str(e)}")
            MARKET_HOURS['holidays'] = []

# Tatil günlerini yükle
load_market_holidays() 