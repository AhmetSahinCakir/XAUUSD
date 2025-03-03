import time
from utils.mt5_connector import MT5Connector
from utils.data_processor import DataProcessor
from utils.risk_manager import RiskManager
from models.lstm_model import LSTMPredictor
from models.rl_model import ForexTradingEnv, RLTrader
import torch
import numpy as np
import os
import pandas as pd
from config import MT5_CONFIG, TRADING_CONFIG, MODEL_CONFIG, DATA_CONFIG
import logging
from datetime import datetime, timedelta
import json
import gc  # Garbage Collector

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

class XAUUSDTradingBot:
    def __init__(self):
        """Initializes the Trading Bot"""
        logger.info("Initializing XAUUSD Trading Bot...")
        print("Initializing XAUUSD Trading Bot...")
        
        # Kullanıcıya modeli yeniden eğitme seçeneği sun
        retrain_option = input("Modelleri yeniden eğitmek istiyor musunuz? (y/n): ").strip().lower()
        self.force_retrain = False
        self.clear_existing_models = False
        
        if retrain_option == 'y':
            self.force_retrain = True
            clear_option = input("Mevcut modelleri silip sıfırdan başlamak istiyor musunuz? (y/n): ").strip().lower()
            if clear_option == 'y':
                self.clear_existing_models = True
                print("Mevcut modeller silinecek ve eğitim sıfırdan başlayacak.")
            else:
                print("Mevcut modeller üzerine eğitim yapılacak.")
        else:
            print("Mevcut modeller kullanılacak (varsa).")
        
        # Initialize MT5 connection with config
        self.mt5 = MT5Connector(
            login=MT5_CONFIG['login'],
            password=MT5_CONFIG['password'],
            server=MT5_CONFIG['server']
        )
        
        # Establish connection first
        if not self.mt5.connect():
            error_msg = "Could not connect to MT5. Please check your credentials and MT5 terminal."
            logger.error(error_msg)
            raise ConnectionError(error_msg)
        
        print("Successfully connected to MT5!")
        logger.info("Successfully connected to MT5")
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            initial_balance=TRADING_CONFIG['initial_balance'],
            risk_per_trade=TRADING_CONFIG['risk_per_trade'],
            max_daily_loss=TRADING_CONFIG['max_daily_loss']
        )
        
        # Set up timeframes
        self.timeframes = DATA_CONFIG['timeframes']
        
        # Load or create models
        self.load_or_create_models()
        
        # Check if models need retraining
        self.check_and_retrain_models()
        
        # Trading parameters from config
        self.current_positions = {}
        
    def check_and_retrain_models(self):
        """Check if models need retraining and retrain if necessary"""
        # Create metadata file path
        metadata_file = "saved_models/training_metadata.json"
        
        # Default values if no metadata exists
        last_training_time = None
        retraining_interval_days = 7  # Retrain weekly by default
        
        # Check if metadata exists
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    last_training_time = datetime.fromisoformat(metadata.get('last_training_time', ''))
                    retraining_interval_days = metadata.get('retraining_interval_days', 7)
            except Exception as e:
                logger.error(f"Error reading training metadata: {str(e)}")
                last_training_time = None
        
        # Determine if retraining is needed
        should_retrain = False
        
        # Kullanıcı isterse her zaman yeniden eğit
        if self.force_retrain:
            logger.info("Kullanıcı isteği üzerine modeller yeniden eğitilecek.")
            should_retrain = True
        elif last_training_time is None:
            logger.info("No previous training metadata found. Will train models.")
            should_retrain = True
        else:
            time_since_last_training = datetime.now() - last_training_time
            if time_since_last_training.days >= retraining_interval_days:
                logger.info(f"Models were trained {time_since_last_training.days} days ago. Retraining needed.")
                should_retrain = True
            else:
                logger.info(f"Models were trained {time_since_last_training.days} days ago. No retraining needed.")
        
        # Retrain if needed
        if should_retrain:
            logger.info("Starting model retraining...")
            self.train_models()
            
            # Update metadata
            try:
                os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump({
                        'last_training_time': datetime.now().isoformat(),
                        'retraining_interval_days': retraining_interval_days
                    }, f)
                logger.info("Updated training metadata.")
            except Exception as e:
                logger.error(f"Error saving training metadata: {str(e)}")

    def load_or_create_models(self):
        """Loads or creates LSTM and RL models"""
        try:
            # Mevcut modelleri silme seçeneği
            if self.clear_existing_models:
                print("Mevcut modelleri silme işlemi başlatılıyor...")
                if os.path.exists('saved_models'):
                    # Modelleri sil
                    lstm_files = [f for f in os.listdir('saved_models') if f.startswith('lstm_model_') and f.endswith('.pth')]
                    rl_files = [f for f in os.listdir('saved_models') if f.startswith('rl_model_') and f.endswith('.zip')]
                    
                    for file in lstm_files + rl_files:
                        try:
                            os.remove(os.path.join('saved_models', file))
                            print(f"Silindi: {file}")
                        except Exception as e:
                            print(f"Dosya silinemedi: {file}, hata: {str(e)}")
                    
                    # Metadata dosyasını da sil
                    if os.path.exists('saved_models/training_metadata.json'):
                        os.remove('saved_models/training_metadata.json')
                        print("Training metadata dosyası silindi.")
                    
                    print("Mevcut modeller silindi.")
                else:
                    print("Silinecek model bulunamadı.")
                
                # Sıfırdan modeller oluştur
                self.lstm_model = LSTMPredictor(
                    input_size=MODEL_CONFIG['lstm']['input_size'],
                    hidden_size=MODEL_CONFIG['lstm']['hidden_size'],
                    num_layers=MODEL_CONFIG['lstm']['num_layers'],
                    dropout=MODEL_CONFIG['lstm']['dropout']
                )
                self.rl_trader = None
                logger.info("Yeni modeller oluşturuldu, eğitim sıfırdan başlayacak.")
                return
            
            # Check for saved models
            lstm_files = [f for f in os.listdir('saved_models') if f.startswith('lstm_model_') and f.endswith('.pth')]
            rl_files = [f for f in os.listdir('saved_models') if f.startswith('rl_model_') and f.endswith('.zip')]
            
            if lstm_files and rl_files:
                # Sort by timestamp (newest first)
                lstm_files.sort(reverse=True)
                rl_files.sort(reverse=True)
                
                # Load the newest models
                lstm_path = os.path.join('saved_models', lstm_files[0])
                rl_path = os.path.join('saved_models', rl_files[0])
                
                # Load LSTM model
                self.lstm_model = LSTMPredictor.load_model(lstm_path)
                logger.info(f"Loaded LSTM model from {lstm_path}")
                
                # Load initial data for RL environment
                initial_data_dict = {}
                for timeframe, num_candles in DATA_CONFIG['default_candles'].items():
                    data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
                    if data is not None and len(data) >= num_candles * 0.8:  # At least 80% data required
                        initial_data_dict[timeframe] = data
                
                # Combine all data
                if initial_data_dict:
                    initial_data = pd.concat(initial_data_dict.values())
                else:
                    logger.warning("Could not get enough initial data")
                    initial_data = pd.DataFrame({
                        'open': [], 'high': [], 'low': [], 'close': [], 'tick_volume': []
                    })
                
                # Set up environment parameters
                env_params = {
                    'df': initial_data,
                    'lstm_model': self.lstm_model,
                    'initial_balance': TRADING_CONFIG['initial_balance'],
                    'max_position_size': 1.0,
                    'transaction_fee': TRADING_CONFIG['transaction_fee']
                }
                
                # Load RL model
                self.rl_trader = RLTrader(lstm_model=self.lstm_model, env_params=env_params)
                self.rl_trader.load(rl_path)
                logger.info(f"Loaded RL model from {rl_path}")
                
            else:
                logger.info("No saved models found. Creating new models...")
                # Create new models
                self.lstm_model = LSTMPredictor(
                    input_size=MODEL_CONFIG['lstm']['input_size'],
                    hidden_size=MODEL_CONFIG['lstm']['hidden_size'],
                    num_layers=MODEL_CONFIG['lstm']['num_layers'],
                    dropout=MODEL_CONFIG['lstm']['dropout']
                )
                
                # Get initial data for RL environment
                initial_data_dict = {}
                for timeframe, num_candles in DATA_CONFIG['default_candles'].items():
                    data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
                    if data is not None and len(data) >= num_candles * 0.8:  # At least 80% data required
                        initial_data_dict[timeframe] = data
                
                # Combine all data
                if initial_data_dict:
                    initial_data = pd.concat(initial_data_dict.values())
                else:
                    logger.warning("Could not get enough initial data")
                    initial_data = pd.DataFrame({
                        'open': [], 'high': [], 'low': [], 'close': [], 'tick_volume': []
                    })
                
                # Set up environment parameters
                env_params = {
                    'df': initial_data,
                    'lstm_model': self.lstm_model,
                    'initial_balance': TRADING_CONFIG['initial_balance'],
                    'max_position_size': 1.0,
                    'transaction_fee': TRADING_CONFIG['transaction_fee']
                }
                
                # Create RL model
                self.rl_trader = RLTrader(lstm_model=self.lstm_model, env_params=env_params)
                
                # Train models
                self.train_models()
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
    def train_models(self):
        """Trains LSTM and RL models"""
        logger.info("Starting model training")
        print("Starting model training...")
        
        try:
            # Get different amounts of data for each timeframe
            train_data_dict = {}
            
            # Use larger datasets for training
            training_candles = {
                "1m": 10000,
                "5m": 5000,
                "15m": 2000
            }
            
            for timeframe, num_candles in training_candles.items():
                data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
                if data is not None and len(data) >= num_candles * 0.8:  # At least 80% data required
                    train_data_dict[timeframe] = data
                    logger.info(f"Collected {len(data)} candles for {timeframe} timeframe")
                    
            # Check if we have enough data
            if not train_data_dict:
                error_msg = "Not enough training data available!"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Combine all data
            all_data = pd.concat(train_data_dict.values())
            logger.info(f"Total training data size: {len(all_data)} rows")
                    
            # Prepare data and add indicators
            try:
                # Make sure we don't have missing values in the original data
                all_data = all_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                # Prepare sequences for LSTM
                sequences, targets = self.data_processor.prepare_data(all_data)
                
                if len(sequences) == 0 or len(targets) == 0:
                    error_msg = "Could not prepare training sequences!"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            except Exception as e:
                logger.error(f"Error during data preparation: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                raise
                
            # Split data
            split_idx = int(len(sequences) * 0.8)
            train_sequences = sequences[:split_idx]
            train_targets = targets[:split_idx]
            val_sequences = sequences[split_idx:]
            val_targets = targets[split_idx:]
            
            # Train LSTM
            logger.info("Training LSTM model...")
            self.lstm_model.train_model(
                train_sequences, train_targets,
                val_sequences, val_targets,
                epochs=50, batch_size=32,
                learning_rate=0.0005
            )
            
            # Save LSTM model manually to enforce consistent naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            lstm_path = f"saved_models/lstm_model_{timestamp}.pth"
            os.makedirs("saved_models", exist_ok=True)
            
            torch.save({
                'model_state_dict': self.lstm_model.state_dict(),
                'hidden_size': self.lstm_model.hidden_size,
                'num_layers': self.lstm_model.num_layers
            }, lstm_path)
            logger.info(f"LSTM model saved to {lstm_path}")
            
            # Check if tech indicators exist, add if needed
            required_cols = ['RSI', 'MACD', 'Signal_Line', 'ATR', 'Upper_Band', 'Lower_Band', 'MA20']
            missing_cols = [col for col in required_cols if col not in all_data.columns]
            
            if missing_cols:
                logger.warning(f"Missing technical indicators detected: {missing_cols}")
                all_data = self.data_processor.add_technical_indicators(all_data)
                
                # After adding, recheck if any still missing
                missing_cols = [col for col in required_cols if col not in all_data.columns]
                if missing_cols:
                    logger.warning(f"Still missing indicators after technical calculation: {missing_cols}")
                    for col in missing_cols:
                        if col == 'RSI':
                            all_data[col] = 50  # Neutral RSI
                        elif col in ['MACD', 'Signal_Line']:
                            all_data[col] = 0
                        elif col == 'ATR':
                            all_data[col] = all_data['close'] * 0.01  # 1% of price
                        elif col in ['Upper_Band', 'Lower_Band']:
                            if 'MA20' in all_data.columns:
                                std = all_data['close'].rolling(window=20).std().fillna(0)
                                if col == 'Upper_Band':
                                    all_data[col] = all_data['MA20'] + (2 * std)
                                else:
                                    all_data[col] = all_data['MA20'] - (2 * std)
                            else:
                                all_data['MA20'] = all_data['close'].rolling(window=20).mean().fillna(all_data['close'])
                                std = all_data['close'].rolling(window=20).std().fillna(0)
                                if col == 'Upper_Band':
                                    all_data[col] = all_data['MA20'] + (2 * std)
                                else:
                                    all_data[col] = all_data['MA20'] - (2 * std)
            
            # Setup environment for RL training
            env_params = {
                'df': all_data,
                'lstm_model': self.lstm_model,
                'initial_balance': TRADING_CONFIG['initial_balance'],
                'max_position_size': 1.0,
                'transaction_fee': TRADING_CONFIG['transaction_fee']
            }
            
            # Initialize RL trainer
            # Daima yeni bir RLTrader oluşturalım çünkü env parametreleri değişmiş olabilir
            logger.info("Initializing RL trainer...")
            try:
                # Önce tüm teknık göstergelerin eklendiğinden emin olalım
                all_data = self.data_processor.add_technical_indicators(all_data)
                
                # RLTrader oluştur
                self.rl_trader = RLTrader(lstm_model=self.lstm_model, env_params=env_params)
                
                # RL model eğitimi
                logger.info("Training RL model...")
                self.rl_trader.train(all_data, total_timesteps=50000)
                
                # Save RL model
                rl_path = f"saved_models/rl_model_{timestamp}.zip"
                self.rl_trader.save(rl_path)
                logger.info(f"RL model saved to {rl_path}")
                
                # Update training metadata
                metadata = {
                    'last_training_time': datetime.now().isoformat(),
                    'retraining_interval_days': 7,
                    'lstm_model_path': lstm_path,
                    'rl_model_path': rl_path
                }
                
                with open("saved_models/training_metadata.json", 'w') as f:
                    json.dump(metadata, f)
                
                logger.info("Model training completed successfully")
                print("Model training completed successfully!")
            except Exception as e:
                logger.error(f"RL training error: {str(e)}")
                # RL hatası nedeniyle durdurma yapmayalım, LSTM modelimiz hala kullanılabilir
                print(f"Warning: RL model training failed, but LSTM model was saved successfully.")
                
                # Update training metadata with at least the LSTM model
                metadata = {
                    'last_training_time': datetime.now().isoformat(),
                    'retraining_interval_days': 7,
                    'lstm_model_path': lstm_path
                }
                
                with open("saved_models/training_metadata.json", 'w') as f:
                    json.dump(metadata, f)
        
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
    def get_trading_signals(self, timeframe):
        """Generates trading signals for each timeframe"""
        try:
            # Get different amounts of data for each timeframe
            candle_counts = {
                "1m": 5000,   # Increased for better ATR calculation
                "5m": 2000,   # Increased for better ATR calculation
                "15m": 1000   # Increased for better ATR calculation
            }
            
            num_candles = candle_counts.get(timeframe, 2000)  # Default to 2000 candles
            print(f"\nFetching {num_candles} candles for {timeframe} timeframe...")
            
            data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
            
            if data is None:
                print(f"Error: Could not fetch data for {timeframe} timeframe")
                return None, None
                
            if len(data) < 100:  # Minimum required for technical indicators
                print(f"Error: Not enough data points for {timeframe} timeframe (got {len(data)}, need at least 100)")
                return None, None
                
            # Print data summary for debugging
            print(f"Received {len(data)} candles for {timeframe} timeframe")
            print(f"Data range: {data['time'].min()} to {data['time'].max()}")
                
            # Prepare data and add technical indicators
            try:
                # Skip add_technical_indicators here, it will be called in prepare_data
                sequences, _ = self.data_processor.prepare_data(data)
                
                if len(sequences) == 0:
                    print(f"Error: Could not prepare LSTM sequences for {timeframe}")
                    return None, None
                
                # Add indicators for the RL model
                data = self.data_processor.add_technical_indicators(data)
                
                if 'ATR' not in data.columns or data['ATR'].isnull().all():
                    print(f"Error: ATR calculation failed for {timeframe} timeframe")
                    return None, None
                
                # Check for NaN values in key columns
                nan_cols = [col for col in data.columns if data[col].isnull().any()]
                if nan_cols:
                    print(f"Warning: NaN values found in columns: {nan_cols}")
                    # Fill NaN values
                    data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
                print(f"Successfully calculated indicators for {timeframe} timeframe")
                
            except Exception as e:
                print(f"Error calculating technical indicators for {timeframe}: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
                
            # LSTM prediction
            try:
                lstm_prediction = self.lstm_model.predict(sequences[-1].unsqueeze(0))
                
                # Check if prediction is valid
                if torch.isnan(lstm_prediction).any() or torch.isinf(lstm_prediction).any():
                    print(f"Error: Invalid LSTM prediction for {timeframe}")
                    return None, None
                
                # Tensor'u float'a çevir
                lstm_prediction_value = lstm_prediction.item()
                    
            except Exception as e:
                print(f"Error in LSTM prediction for {timeframe}: {str(e)}")
                return None, None
            
            # RL prediction for current state
            try:
                # Pass the latest row for RL state preparation
                current_state = self.data_processor.prepare_rl_state(data.iloc[-1])
                
                # Check if state is valid
                if torch.isnan(current_state).any():
                    print(f"Error: Invalid RL state for {timeframe} (contains NaN)")
                    return None, None
                
                # Check observation space dimension and adjust if necessary
                if len(current_state.shape) == 1:
                    current_state = current_state.unsqueeze(0)  # Add batch dimension
                
                # Get RL action
                rl_action, _ = self.rl_trader.predict(current_state)
                
                print(f"Generated predictions for {timeframe} - LSTM: {lstm_prediction_value:.2f}, RL: {rl_action[0]}")
                return lstm_prediction_value, rl_action[0]
                
            except Exception as e:
                print(f"Error in RL prediction for {timeframe}: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None
                
        except Exception as e:
            print(f"Error in get_trading_signals for {timeframe}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
        
    def execute_trade(self, timeframe, lstm_pred, rl_action):
        """Execute trades based on model predictions"""
        try:
            print(f"\n====== TİCARET YÜRÜTME BAŞLANGIÇ ({timeframe}) ======")
            # Get latest data for trade execution
            data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=100)
            
            if data is None or len(data) < 30:  # Minimum data requirement
                print(f"Uyarı: {timeframe} için yeterli veri yok, işlem iptal edildi")
                print("====== TİCARET YÜRÜTME BİTİŞ ======\n")
                return
                
            # Add technical indicators
            data = self.data_processor.add_technical_indicators(data)
            
            # Check if ATR is calculated correctly
            if 'ATR' not in data.columns or data['ATR'].isnull().all() or data['ATR'].iloc[-1] == 0:
                print(f"Uyarı: {timeframe} için ATR değeri hesaplanamadı, işlem iptal edildi")
                print("====== TİCARET YÜRÜTME BİTİŞ ======\n")
                return
                
            current_price = data['close'].iloc[-1]
            atr_value = data['ATR'].iloc[-1]
            
            # Print ATR value for debugging
            print(f"Mevcut fiyat: {current_price}, ATR: {atr_value}")
            
            # Determine trade direction based on models
            # LSTM prediction > 0.55 indicates up, < 0.45 indicates down
            # RL action: 0 = hold, 1 = buy, 2 = sell
            trade_allowed = False
            trade_type = None
            
            if rl_action == 1 and lstm_pred > 0.55:  # Both models agree on BUY
                trade_type = "BUY"
                trade_allowed = True
                print(f"ALIM sinyali: LSTM ({lstm_pred:.2f}) ve RL ({rl_action}) aynı yönde")
            elif rl_action == 2 and lstm_pred < 0.45:  # Both models agree on SELL
                trade_type = "SELL"
                trade_allowed = True
                print(f"SATIM sinyali: LSTM ({lstm_pred:.2f}) ve RL ({rl_action}) aynı yönde")
            else:
                print(f"İşlem sinyali yok: LSTM ({lstm_pred:.2f}) ve RL ({rl_action}) uyumsuz")
                
            # Execute trade if both models agree and risk management allows
            if trade_allowed and self.risk_manager.can_trade():
                # ATR değerini doğrula
                if atr_value <= 0 or np.isnan(atr_value):
                    print(f"Uyarı: Geçersiz ATR değeri ({atr_value}), varsayılan değer kullanılıyor")
                    atr_value = current_price * 0.01  # Fiyatın %1'i
                
                # MT5'den sembol bilgilerini al
                symbol_info = self.mt5.symbol_info("XAUUSD")
                if symbol_info is None:
                    print("Uyarı: XAUUSD sembol bilgisi alınamadı, işlem iptal edildi")
                    print("====== TİCARET YÜRÜTME BİTİŞ ======\n")
                    return
                
                # Pip değeri (point) ve digits değeri
                point = symbol_info.point
                digits = symbol_info.digits
                
                # SL/TP için asgari mesafeler (genellikle 20-50 pip)
                min_sl_distance = point * 50
                min_tp_distance = point * 50
                
                # SL/TP hesaplama
                if trade_type == "BUY":
                    # BUY için SL, entry'nin altında olmalı
                    sl_raw = current_price - (atr_value * 1.5)
                    sl_distance = current_price - sl_raw
                    
                    # Minimum mesafeyi kontrol et
                    if sl_distance < min_sl_distance:
                        sl_raw = current_price - min_sl_distance
                        print(f"SL mesafesi çok yakın, minimum {min_sl_distance} değerine ayarlandı")
                    
                    # TP, entry'nin üzerinde olmalı, R:R oranına göre
                    tp_distance = sl_distance * 2  # 1:2 risk-ödül oranı
                    tp_raw = current_price + tp_distance
                else:  # SELL
                    # SELL için SL, entry'nin üzerinde olmalı
                    sl_raw = current_price + (atr_value * 1.5)
                    sl_distance = sl_raw - current_price
                    
                    # Minimum mesafeyi kontrol et
                    if sl_distance < min_sl_distance:
                        sl_raw = current_price + min_sl_distance
                        print(f"SL mesafesi çok yakın, minimum {min_sl_distance} değerine ayarlandı")
                    
                    # TP, entry'nin altında olmalı, R:R oranına göre
                    tp_distance = sl_distance * 2  # 1:2 risk-ödül oranı
                    tp_raw = current_price - tp_distance
                
                # Fiyatları yuvarlama
                stop_loss = round(sl_raw, digits)
                take_profit = round(tp_raw, digits)
                
                # Print SL/TP details
                print(f"Stop Loss: {stop_loss} (mesafe: {abs(current_price - stop_loss):.5f})")
                print(f"Take Profit: {take_profit} (mesafe: {abs(current_price - take_profit):.5f})")
                
                # Calculate position size - hesap bakiyesinin %1'ini riske et
                account_info = self.mt5.get_account_info()
                if account_info:
                    balance = account_info.balance
                else:
                    print("Hesap bilgisi alınamadı, varsayılan bakiye kullanılıyor")
                    balance = TRADING_CONFIG['initial_balance']
                risk_amount = balance * 0.01  # Hesap bakiyesinin %1'i
                
                # Risk miktarını hesapla (stop loss mesafesine göre)
                point_value = 0.01  # Gold için 1 pip = 0.01 USD (100 oz lot başına)
                pip_risk = abs(current_price - stop_loss) / point
                
                # Pozisyon büyüklüğü (lot) = risk_amount / (pip_risk * point_value * 100)
                # 100 = 1 lot (100 oz altın)
                position_size = risk_amount / (pip_risk * point_value * 100)
                
                # Lot büyüklüğü sınırları
                min_lot = 0.01
                max_lot = 1.0
                position_size = max(min_lot, min(round(position_size, 2), max_lot))
                
                print(f"Hesaplanan lot: {position_size} (Bakiye: ${balance}, Risk: ${risk_amount})")
                print(f"{trade_type} emri yürütülüyor...")
                
                # MT5 bağlantısını kontrol et
                if not self.mt5.connected:
                    print("MT5 bağlı değil, bağlanmaya çalışılıyor...")
                    if not self.mt5.connect():
                        print("MT5'e bağlanılamadı, emir iptal edildi")
                        print("====== TİCARET YÜRÜTME BİTİŞ ======\n")
                        return
                
                # Place order
                order_result = self.mt5.place_order(
                    symbol="XAUUSD",
                    order_type=trade_type,
                    volume=position_size,
                    price=None,  # Market emri için fiyat belirtme
                    sl=stop_loss,
                    tp=take_profit
                )
                
                if order_result:
                    print(f"İşlem başarıyla gerçekleşti!")
                    # Trade sonucunu kaydet
                    self.risk_manager.update_balance(0)  # İşlem başlangıçta sıfır kar/zarar ile başlar
                else:
                    print(f"İşlem gerçekleştirilemedi!")
            elif not trade_allowed:
                print(f"{timeframe} için ticaret sinyali yok")
            elif not self.risk_manager.can_trade():
                print(f"Risk yönetimi {timeframe} için işleme izin vermiyor")
            
            print("====== TİCARET YÜRÜTME BİTİŞ ======\n")
                
        except Exception as e:
            print(f"{timeframe} için işlem hatası: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error information
            print("====== TİCARET YÜRÜTME HATA ======\n")
        
    def run(self):
        """Runs the Trading Bot"""
        print("\nStarting XAUUSD Trading Bot...")
        print("Press Ctrl+C to stop the bot")
        print("==============================")
        logger.info("Starting XAUUSD Trading Bot")
        
        # Memory usage monitoring variables
        last_gc_time = time.time()
        gc_interval = 600  # Run garbage collection every 10 minutes
        
        try:
            while True:
                if not self.mt5.connect():
                    logger.error("MT5 connection failed, retrying...")
                    print("MT5 connection failed, retrying...")
                    time.sleep(60)
                    continue
                
                current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n[{current_time}]")
                logger.info(f"Processing cycle at {current_time}")
                
                # Check trades for each timeframe
                for timeframe in self.timeframes:
                    try:
                        lstm_pred, rl_action = self.get_trading_signals(timeframe)
                        if lstm_pred is not None and rl_action is not None:
                            self.execute_trade(timeframe, lstm_pred, rl_action)
                    except Exception as e:
                        logger.error(f"Error processing {timeframe}: {str(e)}")
                        print(f"Error processing {timeframe}: {str(e)}")
                        continue  # Continue with next timeframe
                        
                # Check and reset daily statistics
                account_info = self.mt5.get_account_info()
                if account_info:
                    balance_msg = f"Balance: ${account_info.balance:.2f}, Daily P/L: ${self.risk_manager.daily_pnl:.2f}"
                    print(balance_msg)
                    logger.info(balance_msg)
                    
                # Reset statistics at the start of each day
                if time.localtime().tm_hour == 0 and time.localtime().tm_min == 0:
                    self.risk_manager.reset_daily_stats()
                    logger.info("Daily statistics reset")
                    print("Daily statistics reset")
                    
                # Memory management - run garbage collection periodically
                if time.time() - last_gc_time > gc_interval:
                    logger.info("Running memory cleanup")
                    gc.collect()  # Force garbage collection
                    last_gc_time = time.time()
                    mem_info = f"Memory usage: {self.get_memory_usage():.2f} MB"
                    logger.info(mem_info)
                    print(f"Memory cleanup completed. {mem_info}")
                    
                # Wait before next iteration
                logger.info("Waiting for next cycle")
                time.sleep(60)  # Wait for 1 minute
                
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            print("\nBot stopped by user.")
            self.mt5.disconnect()
        except Exception as e:
            logger.error(f"Critical error: {str(e)}")
            print(f"\nError occurred: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.mt5.disconnect()
            
    def get_memory_usage(self):
        """Return current memory usage in MB"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

def check_data_sizes(mt5_connector):
    """
    Checks and compares data sizes for different timeframes
    """
    # Test candle counts
    candle_counts = {
        "1m": [2000, 10000, 20000],
        "5m": [1000, 5000, 10000],
        "15m": [500, 2000, 5000]
    }
    
    results = {}
    
    print("\n=== Data Size Check ===")
    print("--------------------------------")
    
    for timeframe in candle_counts.keys():
        print(f"\n{timeframe} Timeframe Results:")
        print("--------------------------------")
        results[timeframe] = {}
        
        for num_candles in candle_counts[timeframe]:
            data = mt5_connector.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
            
            if data is not None:
                actual_size = len(data)
                coverage = (actual_size / num_candles) * 100
                memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # In MB
                
                results[timeframe][num_candles] = {
                    "requested_candles": num_candles,
                    "received_candles": actual_size,
                    "coverage_ratio": coverage,
                    "memory_usage": memory_usage
                }
                
                print(f"\nRequested Candles: {num_candles}")
                print(f"Received Candles: {actual_size}")
                print(f"Coverage Ratio: {coverage:.2f}%")
                print(f"Memory Usage: {memory_usage:.2f} MB")
            else:
                print(f"\nRequested Candles: {num_candles}")
                print("Could not retrieve data!")
    
    return results

def display_status(data_dict):
    """Displays a simple status of the data"""
    print("\n=== System Status ===")
    print("MT5 Connection: ✓")
    print("\nData Overview:")
    for timeframe, data in data_dict.items():
        if data is not None:
            print(f"{timeframe}: {len(data)} candles loaded ✓")
        else:
            print(f"{timeframe}: No data ✗")
    print("==================\n")

def main():
    """Ana program"""
    try:
        print("\n====== ALTIN TİCARET BOTUNA HOŞGELDİNİZ ======")
        print("Bot başlatılıyor...")
        
        # Logger ayarları
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s\n%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger = logging.getLogger('TradingBot')
        
        # MT5 bağlantısı
        print("\nMetaTrader 5'e bağlanılıyor...")
        mt5_connector = MT5Connector()
        
        # Bağlantı kontrolü
        if not mt5_connector.connected:
            print("MT5 bağlantısı kurulamadı! Lütfen MT5 terminalinin açık olduğundan emin olun.")
            print("Program sonlandırılıyor...")
            return
        
        # Hesap bilgilerini göster
        account_info = mt5_connector.get_account_info()
        if account_info:
            print(f"\nBağlantı başarılı!")
            print(f"Hesap: {account_info.login}")
            print(f"Sunucu: {account_info.server}")
            print(f"Bakiye: ${account_info.balance:.2f}")
            print(f"Özsermaye: ${account_info.equity:.2f}")
            print(f"Marjin: ${account_info.margin:.2f}")
            print(f"Serbest Marjin: ${account_info.margin_free:.2f}")
        
        # Sembol bilgilerini kontrol et
        symbol_info = mt5_connector.symbol_info("XAUUSD")
        if symbol_info is None:
            print("\nUyarı: XAUUSD sembolü bulunamadı! Lütfen sembolün doğru olduğundan emin olun.")
            print("Program sonlandırılıyor...")
            return
        
        print(f"\nXAUUSD sembol bilgileri:")
        print(f"Pip değeri: {symbol_info.point}")
        print(f"Spread: {symbol_info.spread} puan")
        print(f"Minimum lot: {symbol_info.volume_min}")
        print(f"Maksimum lot: {symbol_info.volume_max}")
        print(f"Lot adımı: {symbol_info.volume_step}")
        
        # Veri işleyici oluştur
        data_processor = DataProcessor()
        
        # Risk yöneticisi oluştur
        risk_manager = RiskManager(mt5_connector)
        
        # Modelleri oluştur
        lstm_model = LSTMPredictor(data_processor)
        rl_model = RLTrader(data_processor)
        
        # Bot oluştur
        bot = XAUUSDTradingBot()
        
        # Modelleri başlat
        print("\nModeller yükleniyor...")
        try:
            bot.load_or_create_models()
            print("Modeller başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            print("Modeller yüklenemedi, program sonlandırılıyor...")
            return
        
        # Bot'u başlat
        print("\n====== BOT BAŞLATILIYOR ======")
        bot.run()
        
    except KeyboardInterrupt:
        print("\n\nBot kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n\nBot çalışırken beklenmeyen hata: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # MT5 bağlantısını kapat
        try:
            if 'mt5_connector' in locals() and mt5_connector.connected:
                mt5_connector.disconnect()
                print("\nMT5 bağlantısı kapatıldı.")
        except:
            pass
        
        print("\n====== BOT SONLANDIRILDI ======")

if __name__ == "__main__":
    main() 