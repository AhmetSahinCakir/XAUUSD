MT5_CONFIG = {
    'login': 90674420,        # MT5 hesap numaranız (test scriptinden doğrulandı)
    'password': '_mKtFp2z',   # MT5 ana şifrenizi buraya yazın
    'server': 'MetaQuotes-Demo'    # MT5 sunucu adınız (test scriptinden doğrulandı)
}

# Trading Parameters
TRADING_CONFIG = {
    'initial_balance': 10000.0,
    'risk_per_trade': 1.0,    # Risk percentage per trade (1%)
    'max_daily_loss': 5.0,    # Maximum daily loss percentage (5%)
    'transaction_fee': 0.0001, # Transaction fee (0.01%)
}

# Model Parameters
MODEL_CONFIG = {
    'lstm': {
        'input_size': 5,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.3
    },
    'rl': {
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2
    },
    'batch_size': 64,
    'epochs': 100,
    'learning_rate': 0.001,
    'sequence_length': 60,
    'prediction_steps': 1,
    'train_split': 0.8,
    'weight_recent_factor': 2.0,
    'patience': 10
}

# Data Parameters
DATA_CONFIG = {
    'timeframes': ['1m', '5m', '15m'],
    'default_candles': {
        '1m': 5000,
        '5m': 3000,
        '15m': 2000
    },
    'training_candles': {
        '1m': 100000,
        '5m': 50000,
        '15m': 20000
    },
    'retraining_interval_days': 7
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'trading_bot.log',
            'mode': 'a',
        },
    },
    'loggers': {
        'TradingBot': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
        'TradingBot.DataProcessor': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'TradingBot.MT5Connector': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        },
        'TradingBot.LSTMModel': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
} 