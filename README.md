# XAUUSD Trading Bot

This project is an automated trading bot for XAUUSD (Gold) pair. The bot makes trading decisions using LSTM and Reinforcement Learning (RL) models.

## 🚀 Features

- **Multi-Timeframe Analysis**: Simultaneous analysis on 1m, 5m, and 15m charts
- **Hybrid AI Model**: Combination of LSTM and RL models
- **Automated Risk Management**: 
  - Maximum 1% risk per trade
  - Maximum 5% daily loss limit
  - ATR-based dynamic stop loss
  - Take profit based on Risk/Reward ratio
- **MT5 Integration**: Full integration with MetaTrader 5

## 📋 Requirements

- Python 3.8+
- MetaTrader 5
- Required Python libraries (listed in `requirements.txt`)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/username/XAUUSD.git
cd XAUUSD
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install required libraries:
```bash
pip install -r requirements.txt
```

4. Install MetaTrader 5 and create a demo account.

## 💻 Usage

1. Start MetaTrader 5 and log in to your account.

2. Run the bot:
```bash
python main.py
```

When first started, the bot will:
- Create necessary models
- Train models with historical data
- Then start real-time trading

## ⚙️ Configuration

Basic parameters can be adjusted in `main.py`:
- `initial_balance`: Starting balance
- `risk_per_trade`: Risk percentage per trade
- `max_daily_loss`: Maximum daily loss percentage
- `timeframes`: Time periods to analyze

## 📊 Performance Monitoring

While running, the bot prints:
- Current balance
- Daily profit/loss
- Opened trades
- Model predictions
and other information to the console.

## ⚠️ Risk Warning

This bot is experimental and does not constitute financial advice. Before using on a real account:
- Test extensively on a demo account
- Carefully adjust risk management parameters
- Continuously monitor market conditions

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details. 