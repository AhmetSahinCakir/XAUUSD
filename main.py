import time
from utils.mt5_connector import MT5Connector
from utils.data_processor import DataProcessor
from utils.risk_manager import RiskManager
from utils.market_hours import MarketHours
from utils.system_monitor import SystemMonitor
from models.lstm_model import LSTMPredictor
from models.rl_model import ForexTradingEnv, RLTrader
import torch
import numpy as np
import os
import pandas as pd
from config import MT5_CONFIG, TRADING_CONFIG, MODEL_CONFIG, DATA_CONFIG, LOGGING_CONFIG
import logging
import logging.config
from datetime import datetime, timedelta
import json
import gc  # Garbage Collector
import traceback  # Hata ayıklama için traceback modülünü ekle
import signal
import configparser
import sys

# Loglama konfigürasyonunu uygula
logging.config.dictConfig(LOGGING_CONFIG)

# Ana logger'ı oluştur
logger = logging.getLogger("TradingBot")
logger.info("XAUUSD Trading Bot başlatılıyor...")

# Modül loggerları
mt5_logger = logging.getLogger("TradingBot.MT5Connector")
data_logger = logging.getLogger("TradingBot.DataProcessor")
lstm_logger = logging.getLogger("TradingBot.LSTMModel")

class XAUUSDTradingBot:
    def __init__(self):
        """Initializes Trading Bot"""
        # Set up paths and config
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.base_dir, 'config.json')
        
        # Load config
        self.load_config()
        
        # Initialize logger
        self.setup_logger()
        
        # Setup MT5 connector
        self.mt5 = None
        
        # Setup data processor
        self.data_processor = None
        
        # Setup market hours checker
        self.market_hours = None
        
        # Setup system monitor
        self.system_monitor = None
        
        # Model instances
        self.lstm_model = None
        self.lstm_models = {}  # Dictionary to store LSTM models for each timeframe
        self.rl_trader = None
        
        # Data storage
        self.data = {}  # Dictionary to store data for each timeframe
        
        # Timeframes to monitor
        self.timeframes = DATA_CONFIG['timeframes']
        
        # Risk Manager
        self.risk_manager = None
        
        # Parameters from user
        self.retrain_models = False
        self.clear_existing_models = False
        
        # Bot durumu
        self.is_running = False
        
        # Initialize everything
        self.initialize()
        
    def initialize(self):
        """Initializes all components and models"""
        try:
            # Create saved_models directory if it doesn't exist
            os.makedirs('saved_models', exist_ok=True)
            
            logger.info("Initializing XAUUSD Trading Bot...")
            
            # Connect to MT5
            if hasattr(self, 'mt5') and self.mt5 is not None and hasattr(self.mt5, 'connected') and self.mt5.connected:
                print("MT5 zaten bağlı!")
            else:
                print("\n==================================================")
                print("ℹ️ MetaTrader 5'e bağlanılıyor...")
                print("==================================================")
                self.mt5 = MT5Connector()
                if not self.mt5.connect():
                    raise ConnectionError("Could not connect to MT5. Please check if MT5 is running.")
            
            # Initialize SystemMonitor
            self.system_monitor = SystemMonitor(
                mt5_connector=self.mt5,
                emergency_callback=self.handle_emergency
            )
            
            # Show account info
            account_info = self.mt5.get_account_info()
            if account_info:
                print("\n==================================================")
                print("✅ Bağlantı başarılı!")
                print(f"Hesap: {account_info.login}")
                print(f"Sunucu: {account_info.server}")
                print(f"Bakiye: ${account_info.balance:.2f}")
                print(f"Özsermaye: ${account_info.equity:.2f}")
                print(f"Marjin: ${account_info.margin:.2f}")
                print(f"Serbest Marjin: ${account_info.margin_free:.2f}")
                
                # Get symbol info
                symbol_info = self.mt5.symbol_info("XAUUSD")
                if symbol_info:
                    print("\nXAUUSD sembol bilgileri:")
                    print(f"Pip değeri: {symbol_info.point}")
                    print(f"Spread: {symbol_info.spread} puan")
                    print(f"Minimum lot: {symbol_info.volume_min}")
                    print(f"Maksimum lot: {symbol_info.volume_max}")
                    print(f"Lot adımı: {symbol_info.volume_step}")
                print("==================================================")
            
            # Initialize Data Processor
            self.data_processor = DataProcessor()
            
            # Initialize Risk Manager with initial balance from account info
            initial_balance = account_info.balance if account_info else TRADING_CONFIG['initial_balance']
            self.risk_manager = RiskManager(initial_balance=initial_balance)
            
            # Prompt user for model retraining
            self.retrain_models = False
            self.clear_existing_models = False
            
            # Prompt user for retraining
            retrain_input = input("\n==================================================\nModelleri yeniden eğitmek istiyor musunuz? (y/n): ").strip().lower()
            print("==================================================")
            
            if retrain_input == 'y':
                self.retrain_models = True
                clean_start_input = input("\n==================================================\nMevcut modelleri silip sıfırdan başlamak istiyor musunuz? (y/n): ").strip().lower()
                print("==================================================")
                
                if clean_start_input == 'y':
                    self.clear_existing_models = True
                    print("\n==================================================")
                    print("ℹ️ Mevcut modeller silinecek ve eğitim sıfırdan başlayacak.")
                    print("==================================================")
            
            # Load or create models
            self.load_or_create_models(retrain=self.retrain_models, clean_start=self.clear_existing_models)
                
            print("Modeller başarıyla yüklendi.")
            
            # MarketHours nesnesini başlat
            self.market_hours = MarketHours()
            
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            print("\n==================================================")
            print(f"❌ Bot başlatma hatası: {str(e)}")
            print("==================================================")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
    def check_and_retrain_models(self):
        """Check if models need retraining and retrain if necessary"""
        # Create metadata file path
        metadata_file = "saved_models/training_metadata.json"
        
        # Default values if no metadata exists
        last_training_time = None
        retraining_interval_days = DATA_CONFIG['retraining_interval_days']  # Get from config
        
        # Check if metadata exists
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    last_training_time = datetime.fromisoformat(metadata.get('last_training_time', ''))
                    # Still read from metadata but use config as fallback
                    retraining_interval_days = metadata.get('retraining_interval_days', DATA_CONFIG['retraining_interval_days'])
            except Exception as e:
                logger.error(f"Error reading training metadata: {str(e)}")
                last_training_time = None
        
        # Determine if retraining is needed
        should_retrain = False
        
        # Kullanıcı isterse her zaman yeniden eğit
        if self.retrain_models:
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

    def load_or_create_models(self, retrain=False, clean_start=False):
        """
        Modelleri yükler veya yeni modeller oluşturur
        
        Parametreler:
        - retrain: Modelleri yeniden eğit
        - clean_start: Mevcut modelleri sil ve sıfırdan başla
        
        Dönüş:
        - bool: Modeller yüklendiyse True, yüklenemediyse ve eğitim gerekiyorsa False
        """
        try:
            # Model dizinini oluştur
            os.makedirs('saved_models', exist_ok=True)
            
            # Check if we need to retrain
            if retrain:
                logger.info("Kullanıcı isteği üzerine modeller yeniden eğitilecek.")
                # If user wants clean start, delete existing models
                if clean_start:
                    logger.info("Mevcut modeller silinecek ve eğitim sıfırdan başlayacak.")
                    
                    # Delete existing model files
                    deleted_count = 0
                    print("\n==================================================")
                    print("ℹ️ Mevcut modelleri silme işlemi başlatılıyor...")
                    for file in os.listdir('saved_models'):
                        if file.endswith('.pth') or file.endswith('.zip') or file == 'training_metadata.json':
                            os.remove(os.path.join('saved_models', file))
                            deleted_count += 1
                            
                    # Delete metadata file
                    if os.path.exists('saved_models/training_metadata.json'):
                        os.remove('saved_models/training_metadata.json')
                        
                    if deleted_count > 0:
                        print(f"✅ {deleted_count} model dosyası silindi.")
                    else:
                        print("ℹ️ Silinecek model dosyası bulunamadı.")
                    print("==================================================")
                    
                    # Sıfırdan modeller oluştur ama eğitme
                    # Genel LSTM modeli (geriye dönük uyumluluk için)
                    self.lstm_model = LSTMPredictor(
                        input_size=MODEL_CONFIG['lstm']['input_size'],
                        hidden_size=MODEL_CONFIG['lstm']['hidden_size'],
                        num_layers=MODEL_CONFIG['lstm']['num_layers'],
                        dropout=MODEL_CONFIG['lstm']['dropout']
                    )
                    
                    # Her zaman dilimi için ayrı LSTM modelleri
                    for timeframe in self.timeframes:
                        self.lstm_models[timeframe] = LSTMPredictor(
                            input_size=MODEL_CONFIG['lstm']['input_size'],
                            hidden_size=MODEL_CONFIG['lstm']['hidden_size'],
                            num_layers=MODEL_CONFIG['lstm']['num_layers'],
                            dropout=MODEL_CONFIG['lstm']['dropout']
                        )
                    
                    self.rl_trader = None
                    logger.info("Yeni modeller oluşturuldu, eğitim bekliyor.")
                    
                    # Eğitim gerektiğini bildir
                    return False
                else:
                    # Retrain existing models
                    return False
            
            # Check for saved models
            # First check for timeframe-specific models
            timeframe_models_found = False
            loaded_models_count = 0
            
            # Check if metadata exists
            if os.path.exists('saved_models/training_metadata.json'):
                try:
                    with open('saved_models/training_metadata.json', 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if models section exists
                    if 'models' in metadata:
                        # Load models for each timeframe
                        for timeframe, model_info in metadata['models'].items():
                            model_path = model_info.get('model_path')
                            if model_path:
                                # .pth uzantısını kontrol et ve gerekirse ekle
                                if not model_path.endswith('.pth'):
                                    model_path = f"{model_path}.pth"
                                
                                full_path = os.path.join('saved_models', model_path)
                                if os.path.exists(full_path):
                                    try:
                                        # Ayrıntılı yükleme mesajını loglara kaydediyoruz, ama konsola ayrıntı verme
                                        logger.info(f"Loading LSTM model for {timeframe} from {full_path}")
                                        self.lstm_models[timeframe] = LSTMPredictor.load_model(full_path)
                                        timeframe_models_found = True
                                        loaded_models_count += 1
                                    except Exception as e:
                                        logger.error(f"Error loading LSTM model for {timeframe}: {str(e)}")
                                else:
                                    logger.error(f"Model file not found: {full_path}")
                except Exception as e:
                    logger.error(f"Error reading metadata: {str(e)}")
            
            # If no timeframe-specific models found, check for legacy models
            if not timeframe_models_found:
                logger.info("Zaman dilimi bazlı model bulunamadı, eski model formatı kontrol ediliyor...")
                
                # Check if saved_models directory exists
                if not os.path.exists('saved_models'):
                    os.makedirs('saved_models', exist_ok=True)
                    logger.info("Created saved_models directory")
                
                # Check for legacy models
                try:
                    lstm_files = [f for f in os.listdir('saved_models') if f.startswith('lstm_model_') and f.endswith('.pth')]
                    rl_files = [f for f in os.listdir('saved_models') if f.startswith('rl_model_') and f.endswith('.zip')]
                except Exception as e:
                    logger.error(f"Error listing model files: {str(e)}")
                    lstm_files = []
                    rl_files = []
                
                if lstm_files:
                    # Sort by timestamp (newest first)
                    lstm_files.sort(reverse=True)
                    
                    # Load the newest model
                    lstm_path = os.path.join('saved_models', lstm_files[0])
                    
                    try:
                        # Load LSTM model - loglara ayrıntılı kaydet ama konsola ayrıntı verme
                        logger.info(f"Loading legacy LSTM model from {lstm_path}")
                        self.lstm_model = LSTMPredictor.load_model(lstm_path)
                        
                        # Also use this model for all timeframes
                        for timeframe in self.timeframes:
                            self.lstm_models[timeframe] = self.lstm_model
                            logger.info(f"Using legacy LSTM model for {timeframe}")
                            loaded_models_count += 1
                        
                        timeframe_models_found = True
                    except Exception as e:
                        logger.error(f"Error loading legacy LSTM model: {str(e)}")
                    
                    # Load RL model if available
                    if rl_files:
                        rl_files.sort(reverse=True)
                        rl_path = os.path.join('saved_models', rl_files[0])
                        
                        # Load initial data for RL environment
                        initial_data_dict = {}
                        for timeframe, num_candles in DATA_CONFIG['default_candles'].items():
                            data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
                            if data is not None and len(data) >= num_candles * 0.8:
                                initial_data_dict[timeframe] = data
                        
                        # Combine all data
                        if initial_data_dict:
                            initial_data = pd.concat(initial_data_dict.values())
                            
                            # Set up environment parameters
                            env_params = {
                                'df': initial_data,
                                'lstm_model': self.lstm_model,
                                'initial_balance': TRADING_CONFIG['initial_balance'],
                                'max_position_size': 1.0,
                                'transaction_fee': TRADING_CONFIG['transaction_fee']
                            }
                            
                            # Load RL model
                            try:
                                self.rl_trader = RLTrader(lstm_model=self.lstm_model, env_params=env_params)
                                self.rl_trader.load(rl_path)
                                logger.info(f"Loaded RL model from {rl_path}")
                            except Exception as e:
                                logger.error(f"Error loading RL model: {str(e)}")
                                self.rl_trader = None
                
            # Modellerin durumunu rapor et
            if loaded_models_count == len(self.timeframes):
                logger.info("Tüm zaman dilimleri için modeller başarıyla yüklendi")
                print("\n==================================================")
                print("✅ Tüm modeller başarıyla yüklendi.")
                print("==================================================")
                return True
            elif loaded_models_count > 0:
                logger.warning(f"Yalnızca {loaded_models_count}/{len(self.timeframes)} model yüklenebildi")
                print("\n==================================================")
                print(f"⚠️ Uyarı: Yalnızca {loaded_models_count}/{len(self.timeframes)} model yüklenebildi.")
                print("Diğer modeller bulunamadı veya yüklenemedi.")
                print("==================================================")
                return True  # Yine de bazı modeller başarıyla yüklendi
            else:
                logger.error("Hiçbir model yüklenemedi!")
                
                # Hiçbir model bulunamadıysa, yeni boş modeller oluştur ama eğitme
                logger.info("No saved models found. Need to create and train new models.")
                
                # Ana LSTM modeli
                self.lstm_model = LSTMPredictor(
                    input_size=MODEL_CONFIG['lstm']['input_size'],
                    hidden_size=MODEL_CONFIG['lstm']['hidden_size'],
                    num_layers=MODEL_CONFIG['lstm']['num_layers'],
                    dropout=MODEL_CONFIG['lstm']['dropout']
                )
                
                # Her zaman dilimi için LSTM modelleri
                for timeframe in self.timeframes:
                    self.lstm_models[timeframe] = LSTMPredictor(
                        input_size=MODEL_CONFIG['lstm']['input_size'],
                        hidden_size=MODEL_CONFIG['lstm']['hidden_size'],
                        num_layers=MODEL_CONFIG['lstm']['num_layers'],
                        dropout=MODEL_CONFIG['lstm']['dropout']
                    )
                    
                # Eğitim gerektiğini belirt
                return False
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            print("\n==================================================")
            print(f"❌ Model yükleme hatası: {str(e)}")
            print("==================================================")
            return False

    def train_models(self):
        """Train all models (LSTM and RL)"""
        logger.info("Starting model training")
        print("Starting model training...")
        
        try:
            # Get different amounts of data for each timeframe
            # Use larger datasets for training - optimized for better model performance
            training_candles = DATA_CONFIG['training_candles']
            
            logger.info("Starting data collection for training...")
            print("Eğitim için veri toplanıyor. Bu işlem büyük veri miktarı nedeniyle biraz zaman alabilir...")
            
            # Başarılı eğitilen model sayısını takip et
            successfully_trained_count = 0
            total_models = len(self.timeframes)
            
            # Train LSTM models for each timeframe
            for timeframe in self.timeframes:
                print(f"\n{timeframe} zaman dilimi için LSTM modeli eğitiliyor...")
                success = self.train_lstm_model(timeframe)
                if success:
                    print(f"{timeframe} LSTM modeli başarıyla eğitildi.")
                    successfully_trained_count += 1
                else:
                    print(f"{timeframe} LSTM modeli eğitimi başarısız oldu.")
            
            # Update training metadata with retraining interval from config
            metadata = {
                'last_training_time': datetime.now().isoformat(),
                'retraining_interval_days': DATA_CONFIG['retraining_interval_days']
            }
            
            os.makedirs("saved_models", exist_ok=True)
            with open("saved_models/training_metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            # Tüm modeller başarıyla eğitildi mi kontrol et
            if successfully_trained_count == total_models:
                logger.info("Tüm modeller başarıyla eğitildi")
                print("Tüm modeller başarıyla eğitildi!")
            elif successfully_trained_count > 0:
                logger.warning(f"Sadece {successfully_trained_count}/{total_models} model başarıyla eğitildi")
                print(f"Uyarı: {successfully_trained_count}/{total_models} model başarıyla eğitildi. Diğer modeller eğitilemedi.")
            else:
                logger.error("Hiçbir model başarıyla eğitilemedi")
                print("Hata: Hiçbir model başarıyla eğitilemedi! Bot düzgün çalışmayabilir.")
        
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def get_trading_signals(self, timeframe):
        """
        Belirli bir zaman dilimi için ticaret sinyalleri üretir
        
        Parametreler:
        - timeframe: Zaman dilimi (örn. '1m', '5m', '15m')
        
        Dönüş:
        - lstm_prediction: LSTM modeli tahmini
        - rl_action: RL modeli aksiyonu
        """
        try:
            # Zaman dilimi için son verileri al
            logger.debug(f"{timeframe} zaman dilimi için ticaret sinyalleri üretiliyor...")
            
            # Son 100 mum verisini al
            candles = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=100)
            
            if candles is None or len(candles) < 60:
                logger.warning(f"{timeframe} için yeterli veri yok. En az 60 mum gerekli.")
                return None, None
            
            # Veri sütunlarını kontrol et
            required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            missing_columns = [col for col in required_columns if col not in candles.columns]
            
            if missing_columns:
                logger.warning(f"{timeframe} verisinde eksik sütunlar: {missing_columns}")
                logger.debug(f"Mevcut sütunlar: {list(candles.columns)}")
                
                # tick_volume eksikse ve volume varsa, tick_volume olarak kullan
                if 'tick_volume' in missing_columns and 'volume' in candles.columns:
                    logger.debug("'volume' sütunu 'tick_volume' olarak kullanılıyor")
                    candles['tick_volume'] = candles['volume']
                    missing_columns.remove('tick_volume')
                
                # Hala eksik sütun varsa
                if missing_columns:
                    logger.error(f"Kritik sütunlar eksik: {missing_columns}")
                    return None, None
            
            # Teknik göstergeleri ekle
            try:
                candles = self.data_processor.add_technical_indicators(candles)
            except Exception as e:
                logger.error(f"Teknik göstergeler eklenirken hata: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None, None
            
            # Fiyat boşluklarını tespit et
            try:
                candles = self.data_processor.detect_price_gaps(candles)
                if candles is None:
                    logger.error("detect_price_gaps None döndürdü, işlem sonlandırılıyor")
                    return None, None
            except Exception as e:
                logger.debug(f"Fiyat boşlukları tespit edilirken hata: {str(e)}")
                if candles is not None:
                    candles['gap'] = 0
                    candles['gap_size'] = 0
                else:
                    logger.error("candles None değeri, işlem sonlandırılıyor")
                    return None, None
            
            # Seans bilgilerini ekle
            try:
                candles = self.data_processor.add_session_info(candles)
            except Exception as e:
                logger.debug(f"Seans bilgileri eklenirken hata: {str(e)}")
                candles['session_asia'] = 0
                candles['session_europe'] = 0
                candles['session_us'] = 0
            
            # LSTM tahmini için veriyi hazırla
            try:
                # Boş DataFrame kontrolü
                if candles is None or candles.empty:
                    logger.error("LSTM tahmini için hazırlanan veri boş veya None")
                    return None, None
                
                lstm_data = self.data_processor.prepare_prediction_data(candles)
                
                # None kontrolü
                if lstm_data is None:
                    logger.error("prepare_prediction_data None döndürdü")
                    return None, None
                
                # LSTM modelini kullanarak tahmin yap
                lstm_model = self.lstm_models.get(f"lstm_{timeframe}")
                if lstm_model is None:
                    logger.warning(f"{timeframe} için LSTM modeli bulunamadı")
                    return None, None
                
                lstm_prediction = lstm_model.forward(lstm_data)
                lstm_prediction = lstm_prediction.item()
                
                # Tahmin edilen değeri orijinal ölçeğe dönüştür
                lstm_prediction = self.data_processor.inverse_transform_price(lstm_prediction)
                
                # Son kapanış fiyatı
                last_close = candles['close'].iloc[-1]
                
                # Yön ve yüzde değişim
                direction = "YUKARI" if lstm_prediction > last_close else "AŞAĞI"
                change_pct = abs(lstm_prediction - last_close) / last_close * 100
                
                logger.info(f"{timeframe} LSTM Tahmini: {lstm_prediction:.2f} ({direction}, %{change_pct:.2f})")
                
                # RL durumunu hazırla
                rl_state = self.data_processor.prepare_rl_state(candles)
                
                # RL modelini kullanarak aksiyon al
                rl_model = self.rl_trader
                if rl_model is None:
                    logger.debug(f"{timeframe} için RL modeli bulunamadı")
                    rl_action = None
                else:
                    rl_action = rl_model.predict(rl_state)
                    logger.info(f"{timeframe} RL Aksiyonu: {rl_action}")
                
                return lstm_prediction, rl_action
                
            except Exception as e:
                logger.error(f"{timeframe} için tahmin yapılırken hata: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None, None
                
        except Exception as e:
            logger.error(f"{timeframe} için ticaret sinyalleri üretilirken hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
        
    def execute_trade(self, timeframe, lstm_pred, rl_action):
        """Execute trades based on model predictions"""
        try:
            print(f"\n====== TİCARET YÜRÜTME BAŞLANGIÇ ({timeframe}) ======")
            
            # Piyasa açık mı kontrol et
            if not self.market_hours.is_market_open():
                logger.info("Piyasa kapalı, işlem yapılmıyor")
                print("Piyasa kapalı, işlem yapılmıyor")
                return
            
            # MT5 bağlantısını kontrol et
            if not self.mt5.connected:
                logger.error("MT5 bağlantısı yok")
                return
            
            # Güncel fiyat bilgisini al
            current_price = self.mt5.get_current_price("XAUUSD")
            if current_price is None:
                logger.error("Fiyat bilgisi alınamadı")
                return
            
            # Son 100 mumu al
            data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=100)
            if data is None or len(data) < 30:
                logger.error(f"{timeframe} için yeterli veri yok")
                return
            
            # Teknik göstergeleri hesapla
            df = self.data_processor.add_technical_indicators(data)
            if df is None:
                logger.error("Teknik göstergeler hesaplanamadı")
                return
            
            # ATR değerini al
            atr = df['ATR'].iloc[-1]
            if np.isnan(atr) or atr <= 0:
                logger.error("Geçersiz ATR değeri")
                return
            
            print(f"Mevcut fiyat: {current_price}, ATR: {atr}")
            
            # İşlem sinyallerini kontrol et
            if rl_action == 1 and lstm_pred > 0.55:  # ALIM
                trade_type = mt5.ORDER_TYPE_BUY
                print(f"ALIM sinyali: LSTM ({lstm_pred:.2f}) ve RL ({rl_action}) aynı yönde")
            elif rl_action == 2 and lstm_pred < 0.45:  # SATIM
                trade_type = mt5.ORDER_TYPE_SELL
                print(f"SATIM sinyali: LSTM ({lstm_pred:.2f}) ve RL ({rl_action}) aynı yönde")
            else:
                print(f"İşlem sinyali yok: LSTM ({lstm_pred:.2f}) ve RL ({rl_action}) uyumsuz")
                return
            
            # Risk yönetimi kontrolü
            if not self.risk_manager.can_trade():
                logger.warning("Risk yönetimi işleme izin vermiyor")
                return
            
            # Sembol bilgilerini al
            symbol_info = self.mt5.get_symbol_info("XAUUSD")
            if symbol_info is None:
                logger.error("Sembol bilgisi alınamadı")
                return
            
            # Stop loss ve take profit mesafelerini hesapla
            min_stop_level = symbol_info.trade_stops_level * symbol_info.point
            
            if trade_type == mt5.ORDER_TYPE_BUY:
                sl_distance = max(atr * 1.5, min_stop_level)
                tp_distance = sl_distance * 2  # Risk:Reward oranı 1:2
                
                sl_price = current_price - sl_distance
                tp_price = current_price + tp_distance
                price = symbol_info.ask
            else:  # SELL
                sl_distance = max(atr * 1.5, min_stop_level)
                tp_distance = sl_distance * 2  # Risk:Reward oranı 1:2
                
                sl_price = current_price + sl_distance
                tp_price = current_price - tp_distance
                price = symbol_info.bid
            
            # Lot büyüklüğü hesaplama
            account_info = self.mt5.get_account_info()
            if account_info is None:
                logger.error("Hesap bilgisi alınamadı")
                return
            
            risk_amount = account_info.balance * TRADING_CONFIG['risk_per_trade']
            pip_value = symbol_info.trade_tick_value
            pip_distance = abs(price - sl_price) / symbol_info.point
            
            lot_size = round(risk_amount / (pip_distance * pip_value), 2)
            lot_size = max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)
            
            print(f"Stop Loss: {sl_price:.2f} (mesafe: {abs(current_price - sl_price):.2f})")
            print(f"Take Profit: {tp_price:.2f} (mesafe: {abs(current_price - tp_price):.2f})")
            print(f"Hesaplanan lot: {lot_size} (Bakiye: ${account_info.balance}, Risk: ${risk_amount})")
            
            # Emir gönder
            order_result = self.mt5.place_order(
                symbol="XAUUSD",
                order_type=trade_type,
                volume=lot_size,
                price=price,
                sl=sl_price,
                tp=tp_price
            )
            
            if order_result and order_result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"İşlem başarıyla gerçekleşti!")
                self.risk_manager.update_balance(0)  # İşlem başlangıçta sıfır kar/zarar ile başlar
            else:
                error_msg = f"İşlem başarısız: {order_result.comment if order_result else 'Bilinmeyen hata'}"
                logger.error(error_msg)
                print(error_msg)
            
        except Exception as e:
            logger.error(f"İşlem hatası: {str(e)}")
            print(f"İşlem hatası: {str(e)}")
        finally:
            print("====== TİCARET YÜRÜTME BİTİŞ ======\n")
        
    def run(self):
        """Runs the Trading Bot"""
        print("\nStarting XAUUSD Trading Bot...")
        print("Press Ctrl+C to stop the bot")
        print("==============================")
        logger.info("Starting XAUUSD Trading Bot")
        
        # Bot durumunu güncelle
        self.is_running = True
        
        # RL model kontrolü
        if self.rl_trader is None:
            print("\nUYARI: RL modeli bulunamadı veya düzgün yüklenemedi!")
            print("Bot yalnızca LSTM modeli kullanarak çalışacak.")
            print("Daha iyi sonuçlar için, bot'u durdurup modelleri yeniden eğitebilirsiniz.")
            logger.warning("RL model is not initialized, running with LSTM only")
        
        try:
            while self.is_running:
                # Sistem durumunu kontrol et
                self.system_monitor.check()
                system_stats = self.system_monitor.get_system_stats()
                
                # Kritik durum kontrolü
                if system_stats['memory_usage'] > 85:  # %85'den fazla bellek kullanımı
                    logger.warning(f"Yüksek bellek kullanımı: {system_stats['memory_usage']:.1f}%")
                    gc.collect()  # Garbage collection'ı zorla
                
                if not self.mt5.connect():
                    logger.error("MT5 connection failed, retrying...")
                    print("MT5 connection failed, retrying...")
                    time.sleep(60)
                    continue
                
                # Piyasa durumunu kontrol et
                if not self.market_hours.is_market_open():
                    logger.info("Piyasa kapalı, bekleniyor...")
                    print("Piyasa kapalı, bekleniyor...")
                    time.sleep(300)  # 5 dakika bekle
                    continue
                
                # Aktif seansları kontrol et
                active_sessions = self.market_hours.get_current_sessions()
                if active_sessions:
                    logger.info(f"Aktif seanslar: {', '.join(active_sessions)}")
                    print(f"Aktif seanslar: {', '.join(active_sessions)}")
                
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

    def load_config(self):
        """Loads the configuration from config.json"""
        # Config modüllerini bot nesnesinin config özelliğine ata
        self.config = {
            'MT5': MT5_CONFIG,
            'TRADING': TRADING_CONFIG,
            'MODEL': MODEL_CONFIG,
            'DATA': DATA_CONFIG,
            'LOGGING': LOGGING_CONFIG
        }
        logger.info("Konfigürasyon yüklendi")
        
    def setup_logger(self):
        """
        Loglama sistemini yapılandırır
        """
        import logging
        import logging.config
        from config import LOGGING_CONFIG
        
        # Loglama konfigürasyonunu uygula
        logging.config.dictConfig(LOGGING_CONFIG)
        
        # Ana logger'ı al
        self.logger = logging.getLogger("TradingBot")
        
        self.logger.info("Loglama sistemi başlatıldı")
        
        return self.logger

    def train_lstm_model(self, timeframe):
        """
        Belirli bir zaman dilimi için LSTM modelini eğitir
        
        Parametreler:
        - timeframe: Zaman dilimi (örn. '1m', '5m', '15m')
        
        Dönüş:
        - bool: Eğitim başarılı ise True, değilse False
        """
        try:
            logger.info(f"{timeframe} zaman dilimi için LSTM modeli eğitiliyor...")
            
            # Eğitim için gereken mum sayısını al
            training_candles = self.config['DATA']['training_candles'].get(timeframe, 10000)
            
            # Tolerans değeri (gereken veri miktarının %99'u yeterli olsun)
            min_required_candles = int(training_candles * 0.99)
            
            logger.info(f"{timeframe} için gereken mum sayısı: {training_candles} (minimum: {min_required_candles})")
            print(f"Gerekli veri miktarı: {training_candles} mum (minimum: {min_required_candles})")
            
            # Veri toplama
            logger.debug(f"{timeframe} için veri toplanıyor...")
            print(f"Veri toplanıyor ({timeframe})...")
            
            # Bellek kullanımını yönetmek için veriyi parçalar halinde al
            chunk_size = 10000  # Her seferde alınacak mum sayısı
            chunks_needed = (training_candles + chunk_size - 1) // chunk_size
            
            all_data = []
            total_collected = 0
            
            for i in range(chunks_needed):
                # Son chunk için kalan mum sayısını hesapla
                remaining = training_candles - total_collected
                current_chunk = min(chunk_size, remaining)
                
                if current_chunk <= 0:
                    break
                
                # Veriyi al
                chunk_data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=current_chunk, end_date=datetime.now())
                
                if chunk_data is None or len(chunk_data) == 0:
                    logger.warning(f"Chunk {i+1}/{chunks_needed} için veri alınamadı")
                    break
                
                logger.debug(f"Chunk {i+1}/{chunks_needed}: {len(chunk_data)} mum alındı")
                all_data.append(chunk_data)
                total_collected += len(chunk_data)
                
                # Bellek kullanımını logla
                if i % 2 == 0:  # Her 2 chunk'ta bir
                    memory_usage = self.get_memory_usage()
                    logger.debug(f"Bellek kullanımı: {memory_usage:.1f} MB")
            
            # Tüm verileri birleştir
            if not all_data:
                logger.error(f"{timeframe} için veri toplanamadı")
                return False
            
            data = pd.concat(all_data, ignore_index=True)
            logger.info(f"{timeframe} için toplam {len(data)} mum toplandı")
            
            # Yeterli veri var mı kontrol et
            if len(data) < min_required_candles:
                logger.warning(f"{timeframe} için yeterli veri yok. Gereken: {min_required_candles}, Mevcut: {len(data)}")
                
                # Eğer veri miktarı çok az değilse, devam et
                if len(data) < min_required_candles * 0.8:  # %80'inden az ise iptal et
                    logger.error(f"{timeframe} için çok az veri var. Eğitim iptal ediliyor.")
                    return False
                else:
                    logger.warning(f"Veri miktarı yeterli olmasa da eğitime devam ediliyor ({len(data)} mum).")
            
            # Teknik göstergeleri ekle
            logger.debug(f"{timeframe} için teknik göstergeler ekleniyor...")
            data = self.data_processor.add_technical_indicators(data)
            
            # Fiyat boşluklarını tespit et
            logger.debug(f"{timeframe} için fiyat boşlukları tespit ediliyor...")
            data = self.data_processor.detect_price_gaps(data)
            
            # Seans bilgilerini ekle
            logger.debug(f"{timeframe} için seans bilgileri ekleniyor...")
            data = self.data_processor.add_session_info(data)
            
            # Veriyi hazırla (ağırlıklı)
            logger.debug(f"{timeframe} için veri hazırlanıyor...")
            
            # Konfigürasyon parametrelerini al
            sequence_length = self.config['MODEL']['sequence_length']
            train_split = self.config['MODEL']['train_split']
            weight_recent_factor = self.config['MODEL']['weight_recent_factor']
            
            # Veriyi hazırla
            train_seq, train_target, val_seq, val_target, sample_weights = self.data_processor.prepare_data_with_weights(
                data,
                sequence_length=sequence_length,
                train_split=train_split,
                target_column='close',
                prediction_steps=1,
                weight_recent_factor=weight_recent_factor
            )
            
            if train_seq is None or train_target is None:
                logger.error(f"{timeframe} için veri hazırlanamadı")
                return False
            
            logger.info(f"{timeframe} için veri hazırlandı. Eğitim: {train_seq.shape}, Doğrulama: {val_seq.shape if val_seq is not None else 'Yok'}")
            
            # Model parametrelerini al
            input_size = train_seq.shape[2]
            hidden_size = self.config['MODEL']['hidden_size']
            num_layers = self.config['MODEL']['num_layers']
            dropout = self.config['MODEL']['dropout']
            learning_rate = self.config['MODEL']['learning_rate']
            batch_size = self.config['MODEL']['batch_size']
            epochs = self.config['MODEL']['epochs']
            patience = self.config['MODEL']['patience']
            
            # Modeli oluştur
            model = LSTMPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # LSTM modeli eğitim
            logger.info(f"LSTM modeli eğitiliyor...")
            print(f"Model eğitimi başlatılıyor, bu işlem biraz zaman alabilir...")
            
            # Eğitim başlangıç zamanı
            train_start_time = time.time()
            
            # Modeli eğit (verbose=True - ilerleme bilgisi göster)
            model.train_model(
                train_seq,
                train_target,
                val_seq,
                val_target,
                sample_weights=sample_weights,
                batch_size=batch_size, 
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                verbose=True  # İlerleme bilgisi göster
            )
            
            # Eğitim sonu zamanı ve toplam süre
            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            logger.info(f"Eğitim tamamlandı. Süre: {train_duration:.2f} saniye")
            print(f"Eğitim tamamlandı. Süre: {train_duration:.2f} saniye")
            
            # Model kaydetme dizini oluştur
            models_dir = 'saved_models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Model dosya adını oluştur (timeframe_tarih_saat.pth)
            model_filename = f"lstm_model_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = os.path.join(models_dir, f"{model_filename}.pth")
            
            # Modeli kaydet
            saved_path = model.save_checkpoint(model_path)
            logger.info(f"{timeframe} modeli kaydedildi: {saved_path}")
            print(f"{timeframe} modeli kaydedildi: {saved_path}")
            
            # Modeli LSTM modelleri sözlüğüne ekle
            self.lstm_models[timeframe] = model
            
            # Eğitim meta verilerini güncelle (dosya adını uzantısız olarak gönder)
            self.update_training_metadata(timeframe, model_filename)
            
            return True
            
        except Exception as e:
            logger.error(f"{timeframe} modeli eğitilirken hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"{timeframe} modeli eğitimi başarısız oldu: {str(e)}")
            return False

    def update_training_metadata(self, timeframe, model_filename):
        """Update training metadata after model training"""
        try:
            # Create metadata file path
            metadata_file = "saved_models/training_metadata.json"
            os.makedirs("saved_models", exist_ok=True)
            
            # Default metadata
            metadata = {
                'last_training_time': datetime.now().isoformat(),
                'retraining_interval_days': DATA_CONFIG['retraining_interval_days'],
                'models': {}
            }
            
            # Check if metadata exists
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        existing_metadata = json.load(f)
                        # Merge existing metadata
                        metadata.update(existing_metadata)
                        # Ensure models dict exists
                        if 'models' not in metadata:
                            metadata['models'] = {}
                except Exception as e:
                    logger.error(f"Error reading training metadata: {str(e)}")
            
            # Update model info for this timeframe
            if timeframe not in metadata['models']:
                metadata['models'][timeframe] = {}
                
            metadata['models'][timeframe]['last_training_time'] = datetime.now().isoformat()
            metadata['models'][timeframe]['model_path'] = model_filename
            
            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            logger.info(f"Updated training metadata for {timeframe}")
            return True
        except Exception as e:
            logger.error(f"Error updating training metadata: {str(e)}")
            return False

    def save_model(self, model, model_name):
        """Save a model to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs("saved_models", exist_ok=True)
            
            # Save model
            model_path = f"saved_models/{model_name}.pth"
            
            # Save model state
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
                'dropout': model.lstm.dropout if hasattr(model.lstm, 'dropout') else 0.2,
                'timestamp': datetime.now().isoformat()
            }, model_path)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None

    def handle_emergency(self):
        """Acil durum yönetimi"""
        try:
            logger.error("ACİL DURUM: Sistem kritik bir durumla karşılaştı!")
            print("\n==================================================")
            print("⚠️ ACİL DURUM!")
            print("Bot güvenli bir şekilde kapatılıyor...")
            print("==================================================")
            
            emergency_start_time = datetime.now()
            
            # Sistem durumunu logla
            if self.system_monitor:
                stats = self.system_monitor.get_system_stats()
                logger.error(f"Sistem durumu: {stats}")
            
            # Açık pozisyonları kontrol et ve kapat
            if self.mt5 and self.mt5.connected:
                try:
                    # Açık pozisyonları al
                    positions = self.mt5.get_open_positions()
                    if positions:
                        print(f"{len(positions)} açık pozisyon kapatılıyor...")
                        logger.warning(f"{len(positions)} açık pozisyon tespit edildi, kapatılıyor...")
                        
                        for pos in positions:
                            try:
                                # Pozisyonu kapat
                                result = self.mt5.close_position(pos.ticket)
                                
                                if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                                    # Risk yöneticisini güncelle
                                    if self.risk_manager:
                                        pnl = result.profit
                                        self.risk_manager.update_balance(pnl)
                                        logger.info(f"Pozisyon kapatıldı: Ticket {pos.ticket}, P/L: ${pnl:.2f}")
                                else:
                                    error_msg = f"Pozisyon kapatılamadı: Ticket {pos.ticket}"
                                    if result:
                                        error_msg += f", Hata: {result.comment}"
                                    logger.error(error_msg)
                                    print(error_msg)
                                    
                            except Exception as e:
                                logger.error(f"Pozisyon kapatma hatası (Ticket {pos.ticket}): {str(e)}")
                                continue
                except Exception as e:
                    logger.error(f"Açık pozisyonları alma hatası: {str(e)}")
            
            # Risk yöneticisi durumunu kaydet
            if self.risk_manager:
                logger.info(f"Son risk durumu - Günlük P/L: ${self.risk_manager.daily_pnl:.2f}")
            
            # Bağlantıyı kapat
            if self.mt5:
                self.mt5.disconnect()
                logger.info("MT5 bağlantısı güvenli bir şekilde kapatıldı")
            
            # İşlem süresini hesapla
            emergency_duration = (datetime.now() - emergency_start_time).total_seconds()
            logger.error(f"Bot acil durum nedeniyle kapatıldı! (İşlem süresi: {emergency_duration:.2f} saniye)")
            print("\n==================================================")
            print("✅ Tüm işlemler güvenli bir şekilde sonlandırıldı.")
            print("==================================================")
            
            # Programı sonlandır
            os._exit(1)
            
        except Exception as e:
            logger.critical(f"Acil durum yönetimi sırasında kritik hata: {str(e)}")
            logger.critical(traceback.format_exc())
            print("\n==================================================")
            print(f"❌ Acil durum yönetimi başarısız: {str(e)}")
            print("==================================================")
            os._exit(1)

    def stop(self):
        """Botu güvenli bir şekilde durdur"""
        try:
            logger.info("Bot durduruluyor...")
            print("\n==================================================")
            print("Bot durduruluyor, lütfen bekleyin...")
            print("==================================================")
            
            # Bot durumunu güncelle
            self.is_running = False
            
            # Açık pozisyonları kontrol et
            if self.mt5 and self.mt5.connected:
                positions = self.mt5.get_open_positions()
                if positions:
                    print(f"\n{len(positions)} açık pozisyon bulundu.")
                    user_input = input("Açık pozisyonları kapatmak istiyor musunuz? (y/n): ").strip().lower()
                    
                    if user_input == 'y':
                        print("\nPozisyonlar kapatılıyor...")
                        for pos in positions:
                            try:
                                result = self.mt5.close_position(pos.ticket)
                                if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
                                    if self.risk_manager:
                                        self.risk_manager.update_balance(result.profit)
                                    print(f"Pozisyon kapatıldı: Ticket {pos.ticket}, P/L: ${result.profit:.2f}")
                            except Exception as e:
                                logger.error(f"Pozisyon kapatma hatası (Ticket {pos.ticket}): {str(e)}")
                                print(f"Hata: Pozisyon kapatılamadı (Ticket {pos.ticket})")
            
            # Bağlantıyı kapat
            if self.mt5:
                self.mt5.disconnect()
                logger.info("MT5 bağlantısı kapatıldı")
            
            print("\n==================================================")
            print("✅ Bot güvenli bir şekilde durduruldu.")
            print("==================================================")
            
        except Exception as e:
            logger.error(f"Bot durdurulurken hata: {str(e)}")
            print(f"\nHata: Bot durdurulurken bir sorun oluştu: {str(e)}")

def signal_handler(signum, frame):
    """Sinyal yöneticisi"""
    print("\nKapatma sinyali alındı. Bot güvenli bir şekilde kapatılacak...")
    if 'bot' in globals():
        bot.stop()
    sys.exit(0)

def main():
    """Ana program"""
    try:
        # Sinyal yöneticilerini ayarla
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print("\n====== ALTIN TİCARET BOTUNA HOŞGELDİNİZ ======")
        print("Bot başlatılıyor...")

        # Logger ayarları - sadece dosyaya yazsın, konsola değil
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s\n%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename='trading_bot.log',  # Log dosyasına yaz
            filemode='a'                 # Append modu
        )
        logger = logging.getLogger('TradingBot')
        logger.info("XAUUSD Trading Bot başlatılıyor...")
        
        # MT5 bağlantısı
        print("\n==================================================")
        print("ℹ️ MetaTrader 5'e bağlanılıyor...")
        print("==================================================")
        mt5_connector = MT5Connector(
            login=MT5_CONFIG['login'],
            password=MT5_CONFIG['password'],
            server=MT5_CONFIG['server']
        )
        
        # Bağlantı kontrolü
        if not mt5_connector.connected:
            print("\n==================================================")
            print("❌ MT5 bağlantısı kurulamadı!")
            print("Lütfen MT5 terminalinin açık olduğundan emin olun.")
            print("==================================================")
            print("\nProgram sonlandırılıyor...")
            return
        
        # Hesap bilgilerini göster
        account_info = mt5_connector.get_account_info()
        if account_info:
            print("\n==================================================")
            print("✅ Bağlantı başarılı!")
            print(f"Hesap: {account_info.login}")
            print(f"Sunucu: {account_info.server}")
            print(f"Bakiye: ${account_info.balance:.2f}")
            print(f"Özsermaye: ${account_info.equity:.2f}")
            print(f"Marjin: ${account_info.margin:.2f}")
            print(f"Serbest Marjin: ${account_info.margin_free:.2f}")
            print("==================================================")
        
        # Sembol bilgilerini kontrol et
        symbol_info = mt5_connector.symbol_info("XAUUSD")
        if symbol_info is None:
            print("\n==================================================")
            print("⚠️ XAUUSD sembolü bulunamadı!")
            print("Lütfen sembolün doğru olduğundan emin olun.")
            print("==================================================")
            print("\nProgram sonlandırılıyor...")
            return
        
        print("\n==================================================")
        print("ℹ️ XAUUSD sembol bilgileri:")
        print(f"Pip değeri: {symbol_info.point}")
        print(f"Spread: {symbol_info.spread} puan")
        print(f"Minimum lot: {symbol_info.volume_min}")
        print(f"Maksimum lot: {symbol_info.volume_max}")
        print(f"Lot adımı: {symbol_info.volume_step}")
        print("==================================================")
        
        # Veri işleyici oluştur
        data_processor = DataProcessor()
        
        # Risk yöneticisi oluştur
        risk_manager = RiskManager(mt5_connector)
        
        # Bot oluştur
        bot = XAUUSDTradingBot()
        
        # Modelleri başlat
        print("\n==================================================")
        print("ℹ️ Modeller yükleniyor...")
        print("==================================================")
        try:
            models_loaded = bot.load_or_create_models()
            
            # Modeller bulunamadıysa veya yüklenemediyse
            if not models_loaded:
                print("\n==================================================")
                print("⚠️ Model bulunamadı veya yüklenemedi! ⚠️")
                print("Bot çalışmak için eğitilmiş modellere ihtiyaç duyar.")
                print("==================================================")
                
                # Kullanıcıya eğitim isteyip istemediğini sor
                user_input = input("\nModelleri eğitmek ister misiniz? Bu işlem zaman alabilir. (y/n): ").strip().lower()
                
                if user_input == 'y':
                    print("\n==================================================")
                    print("ℹ️ Modeller eğitiliyor... Bu işlem zaman alabilir.")
                    print("==================================================")
                    bot.train_models()
                    print("\n==================================================")
                    print("✅ Modeller eğitildi, bot başlatılıyor...")
                    print("==================================================")
                else:
                    print("\n==================================================")
                    print("⚠️ Modeller eğitilmeden bot düzgün çalışamayacak.")
                    print("==================================================")
                    print("\nProgram sonlandırılıyor...")
                    return
                
        except Exception as e:
            print("\n==================================================")
            print(f"❌ Model yükleme hatası: {str(e)}")
            print("==================================================")
            print("\nModeller yüklenemedi, program sonlandırılıyor...")
            return
        
        # Bot'u başlat
        print("\n==================================================")
        print("✅ BOT BAŞLATILIYOR")
        print("==================================================")
        bot.run()
        
    except KeyboardInterrupt:
        print("\nBot kullanıcı tarafından durduruldu.")
        if 'bot' in locals():
            bot.stop()
    except Exception as e:
        logger.error(f"Kritik hata: {str(e)}")
        print(f"\nKritik hata oluştu: {str(e)}")
        if 'bot' in locals():
            bot.stop()
    finally:
        print("\nProgram sonlandırıldı.")

if __name__ == "__main__":
    main() 