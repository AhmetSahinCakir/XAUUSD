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
    'risk_reward_ratio': float(os.getenv('RISK_REWARD_RATIO', '2.0')),  # Risk/Ödül oranı (default 1:2)
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
        'input_size': 32,          # Giriş özellik sayısı
        'hidden_size': 256,        # LSTM gizli katman boyutu artırıldı
        'num_layers': 3,           # LSTM katman sayısı artırıldı
        'dropout': 0.3,            # Dropout oranı artırıldı
        'bidirectional': True,     # Çift yönlü LSTM
        'threshold_high': 0.65,    # Yüksek tahmin eşiği
        'threshold_low': 0.35,     # Düşük tahmin eşiği
        'gradient_clip': 1.0,      # Gradient clipping değeri artırıldı
    },
    'attention': {
        'dims': [512, 128, 1],     # Attention katman boyutları artırıldı
        'dropout': 0.2             # Attention dropout oranı artırıldı
    },
    'batch_norm': {
        'momentum': 0.1,           # Batch norm momentum
        'eps': 1e-5                # Batch norm epsilon
    },
    'training': {
        'batch_size': 64,          # Mini-batch boyutu artırıldı
        'epochs': 100,             # Epoch sayısı artırıldı
        'learning_rate': 0.001,    # Başlangıç öğrenme oranı artırıldı
        'sequence_length': 60,     # Giriş sekans uzunluğu
        'prediction_steps': 1,     # Tahmin adım sayısı
        'train_split': 0.8,        # Eğitim seti oranı
        'validation_split': 0.1,   # Doğrulama seti oranı
        'test_split': 0.1,         # Test seti oranı
        'early_stopping_patience': 15,  # Erken durdurma sabır sayısı artırıldı
        'reduce_lr_patience': 7,    # LR azaltma sabır sayısı artırıldı
        'reduce_lr_factor': 0.5,   # LR azaltma faktörü
        'min_lr': 1e-6,            # Minimum learning rate
        'weight_decay': 0.001,     # L2 regularizasyon katsayısı azaltıldı
        'loss_scale': 1.0          # Loss ölçeklendirme faktörü artırıldı
    },
    'rl': {
        'window_size': 60,         # Gözlem penceresi
        'total_timesteps': 100000, # Toplam eğitim adımı
        'learning_rate': 0.0001,   # RL öğrenme oranı
        'batch_size': 64,          # RL batch boyutu
        'n_steps': 2048,           # Her güncelleme için adım
        'gamma': 0.99,             # İndirim faktörü
        'policy_kwargs': {
            'net_arch': [256, 128, 64]  # Policy ağ mimarisi
        }
    },
    'integration': {
        'use_lstm_predictions': True,      # LSTM tahminlerini kullan
        'lstm_prediction_weight': 0.7,     # LSTM tahmin ağırlığı
        'reward_lstm_accuracy': True       # Accuracy'yi ödüle dahil et
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