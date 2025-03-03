# MetaTrader 5 Connection Settings
MT5_CONFIG = {
    'login': 90674420,        # MT5 account number
    'password': '_mKtFp2z',   # MT5 main password
    'server': 'MetaQuotes-Demo'    # Default MT5 demo server
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
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2
    },
    'rl': {
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2
    }
}

# Data Parameters
DATA_CONFIG = {
    'timeframes': ['1m', '5m', '15m'],
    'default_candles': {
        '1m': 1000,
        '5m': 1000,
        '15m': 500
    }
} 