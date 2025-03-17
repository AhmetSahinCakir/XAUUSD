import os
import gc
import json
import time
import psutil
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from utils.logger import (
    logger, 
    print_section, 
    print_info, 
    print_warning, 
    print_error, 
    print_success,
    print_balance_info,
    print_trade_info,
    setup_logger
)

# Önce logger'ı kur
setup_logger()

# Diğer modülleri import et
from models.lstm_model import LSTMPredictor
from models.rl_model import RLTrader
from config.config import MODEL_CONFIG, TRADING_CONFIG, MARKET_HOURS, MARKET_CHECK_INTERVALS, DATA_CONFIG, MT5_CONFIG
from utils.mt5_connector import MT5Connector
from utils.data_processor import DataProcessor
from utils.risk_manager import RiskManager
from utils.position_manager import PositionManager
from utils.market_hours import MarketHours
from utils.system_monitor import SystemMonitor
import concurrent.futures
import traceback
import signal
import sys
import codecs
import MetaTrader5 as mt5
import threading

# Windows'da Unicode desteği için
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

class XAUUSDTradingBot:
    def __init__(self):
        """Initializes Trading Bot"""
        # Set up paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create config directory if it doesn't exist
        config_dir = os.path.join(self.base_dir, 'config')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            print_info("Config dizini oluşturuldu")
        
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
        
        # Training state tracking
        self.training_in_progress = False
        self.current_training_model = None
        self.training_interrupted = False
        self.checkpoint_path = None
        self.training_thread = None
        self.training_stop_event = threading.Event()
        
        # Initialize everything
        self.initialize()
        
    def load_config(self):
        """Load configuration from config files"""
        # This is a placeholder method to avoid AttributeError
        # The actual configuration is now loaded from config.config module
        pass
        
    def connect_mt5(self):
        """Connect to MetaTrader 5"""
        try:
            if self.mt5 is None:
                self.mt5 = MT5Connector(
                    login=MT5_CONFIG['login'],
                    password=MT5_CONFIG['password'],
                    server=MT5_CONFIG['server']
                )
            return self.mt5.connected
        except Exception as e:
            print_error(
                f"MT5 bağlantı hatası: {str(e)}",
                f"MT5 connection error: {str(e)}"
            )
            return False
            
    def run(self):
        """Run the trading bot"""
        print_section(
            "BOT ÇALIŞIYOR",
            "BOT IS RUNNING"
        )
        print_info(
            "Bot başarıyla başlatıldı ve çalışıyor.",
            "Bot successfully started and running.",
            "Çıkmak için Ctrl+C tuşlarına basın.",
            "Press Ctrl+C to exit."
        )
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print_info(
                "\nBot kullanıcı tarafından durduruldu.",
                "\nBot stopped by user."
            )
            self.stop()
        except Exception as e:
            print_error(
                f"Bot çalışma hatası: {str(e)}",
                f"Bot runtime error: {str(e)}"
            )
            self.stop()
        
    def run_initial_tests(self):
        """Run initial tests to check if everything is working properly"""
        print_info("Başlangıç testleri çalıştırılıyor...")
        
        tests_passed = True
        
        # MT5 bağlantı testi
        print_info("MT5 bağlantısı test ediliyor...")
        if not self.mt5 or not self.mt5.connected:
            print_error("MT5 bağlantı testi başarısız!")
            tests_passed = False
        else:
            print_success("MT5 bağlantı testi başarılı!")
            
        # Veri alımı testi
        print_info("Veri alımı test ediliyor...")
        try:
            data = self.mt5.get_historical_data("XAUUSD", "5m", num_candles=10)
            if data is None or len(data) < 10:
                print_error("Veri alımı testi başarısız!")
                tests_passed = False
            else:
                print_success("Veri alımı testi başarılı!")
        except Exception as e:
            print_error(f"Veri alımı testi başarısız: {str(e)}")
            tests_passed = False
            
        # Sistem kaynakları testi
        print_info("Sistem kaynakları test ediliyor...")
        memory_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        if memory_available < 1.5:  # 1.5GB minimum gereksinim
            print_warning(f"Kritik düşük bellek! ({memory_available:.1f}GB)")
            print_info("Önerilen minimum bellek: 2GB")
            print_info("Performans sorunları yaşanabilir veya eğitim başarısız olabilir.")
            user_continue = input("Devam etmek istiyor musunuz? (y/n): ").strip().lower()
            if user_continue != 'y':
                return False
        elif memory_available < 2.0:  # 2GB önerilen minimum
            print_warning(f"Düşük bellek: {memory_available:.1f}GB")
            print_info("Önerilen: 2GB veya üzeri")
        else:
            print_success(f"Bellek testi başarılı! ({memory_available:.1f}GB kullanılabilir)")
            
        return tests_passed
        
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
                                
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
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
        
    def initialize(self):
        """Initialize trading bot components"""
        try:
            # Load configs
            self.load_config()
            
            # Initialize MT5 connector
            self.mt5 = MT5Connector(
                login=MT5_CONFIG['login'],
                password=MT5_CONFIG['password'],
                server=MT5_CONFIG['server']
            )
            
            # Connect to MT5 and show account status
            if not self.mt5.connected:
                print_error("MT5 bağlantısı kurulamadı, bot başlatılamıyor!")
                return False
                
            # Get account and symbol info
            account_info = self.mt5.get_account_info()
            symbol_info = self.mt5.symbol_info("XAUUSD")
            
            if account_info is None or symbol_info is None:
                print_error("Hesap veya sembol bilgisi alınamadı!")
                return False
                
            # Show account and connection info
            print_section("HESAP DURUMU")
            print_success(f"MT5 bağlantısı başarılı! ({account_info.server})")
            print_balance_info(f"Hesap: {account_info.login} | Bakiye: ${account_info.balance:,.2f}")
            print_trade_info(f"XAUUSD Spread: {self.mt5.symbol_info('XAUUSD').spread} puan | Lot: {self.mt5.symbol_info('XAUUSD').volume_min:.2f} - {self.mt5.symbol_info('XAUUSD').volume_max:.1f}")
                
            # Initialize data processor
            self.data_processor = DataProcessor(mt5_connector=self.mt5)
            
            # Veri işleme testi
            print_info("Veri işleme testi yapılıyor...")
            test_data = self.mt5.get_historical_data("XAUUSD", "5m", num_candles=100)
            if test_data is None:
                print_error("Test verisi alınamadı!")
                return False
            
            processed_data = self.data_processor.process_data(test_data, "5m")
            if processed_data is None:
                print_error("Veri işleme başarısız!")
                return False
            
            print_success("Veri işleme testi başarılı!")
            
            # Initialize risk manager with account balance
            self.risk_manager = RiskManager(account_info.balance)
            
            # Initialize position manager
            self.position_manager = PositionManager(self.mt5)
            
            # Initialize market hours checker
            self.market_hours = MarketHours()
            
            # System monitor
            self.system_monitor = SystemMonitor(
                mt5_connector=self.mt5,
                emergency_callback=self.handle_emergency
            )
            
            # Başlangıç testlerini sor
            print_section("BAŞLANGIÇ TESTLERİ")
            if input("Başlangıç testlerini çalıştırmak ister misiniz? (y/n): ").strip().lower() == 'y':
                if not self.run_initial_tests():
                    print_warning("Başlangıç testleri başarısız oldu, devam edilecek...")
            
            # Model durumunu sor
            print_section("MODEL DURUMU")
            retrain_input = input("Modelleri eğitmek ister misiniz? (y/n): ").strip().lower()
            
            if retrain_input == 'y':
                print_info("Model eğitimi başlatılıyor...")
                if not self.train_models():
                    print_error("Model eğitimi başarısız! Bot çalışamaz.")
                    return False
            else:
                # Mevcut modelleri yüklemeyi dene
                print_section("MODEL YÜKLEME")
                print_info("Mevcut modeller kontrol ediliyor...")
                if not self.load_or_create_models():
                    print_error("Model yükleme başarısız!")
                    print_info("Bot çalışmak için eğitilmiş modellere ihtiyaç duyar.")
                    
                    while True:
                        print("\nSeçenekler:")
                        print("1) Modelleri eğit")
                        print("2) Çıkış yap")
                        choice = input("\nSeçiminiz (1/2): ").strip()
                        
                        if choice == "1":
                            print_info("Model eğitimi başlatılıyor...")
                            if not self.train_models():
                                print_error("Model eğitimi başarısız! Bot çalışamaz.")
                                return False
                            break
                        elif choice == "2":
                            print_info("Bot kapatılıyor...")
                            return False
                        else:
                            print_warning("Geçersiz seçim! Lütfen 1 veya 2 girin.")
            
            # MarketHours nesnesini başlat
            if not self.market_hours:
                raise Exception("MarketHours başlatılamadı!")
            
            # RL Trader'ı başlat
            try:
                # Veri hazırlığı
                initial_data = self.data_processor.get_latest_data()
                env_params = {
                    'df': initial_data,
                    'window_size': MODEL_CONFIG['rl']['window_size'],
                    'initial_balance': self.risk_manager.initial_balance,
                    'commission': TRADING_CONFIG['commission']
                }
                
                # LSTM modelini al
                lstm_model = self.lstm_models.get('5m')  # 5 dakikalık modeli kullan
                if lstm_model is None:
                    raise Exception("RL Trader için LSTM modeli bulunamadı!")
                
                self.rl_trader = RLTrader(
                    lstm_model=lstm_model,
                    env_params=env_params
                )
                print_success("RL Trader başarıyla başlatıldı!")
            except Exception as e:
                print_warning(f"RL Trader başlatılamadı: {str(e)}")
                print_info("Bot yalnızca LSTM modeli ile devam edecek.")
                self.rl_trader = None
            
            # Start system monitoring
            self.system_monitor.start_monitoring()
            
            print_section("BOT HAZIR!")
            return True
            
        except Exception as e:
            print_error(f"Bot başlatma hatası: {str(e)}")
            logger.error(f"Bot başlatma hatası: {str(e)}")
            logger.error(traceback.format_exc())
            return False

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

    def load_or_create_models(self, retrain=False):
        """Modelleri yükle veya oluştur"""
        try:
            # Model dizinini oluştur
            os.makedirs('saved_models', exist_ok=True)
            
            # Eğer temiz başlangıç isteniyorsa, mevcut modelleri sil
            if self.clear_existing_models:
                print_info("Mevcut modelleri silme işlemi başlatılıyor...")
                model_files = [f for f in os.listdir('saved_models') if f.endswith('.pth')]
                if model_files:
                    for file in model_files:
                        os.remove(os.path.join('saved_models', file))
                    print_success(f"{len(model_files)} model dosyası silindi.")
                else:
                    print_info("Silinecek model dosyası bulunamadı.")
                print_section("MODEL DURUMU")

            # Eğer yeniden eğitim isteniyorsa, direkt eğitime başla
            if retrain:
                logger.info("Model eğitimi başlatılıyor...")
                self.train_models()
                return True

            # Eğer yeniden eğitim istenmiyorsa, mevcut modelleri yüklemeyi dene
            print_section("MODEL YÜKLEME")
            print_info("Mevcut modeller kontrol ediliyor...")
            
            models_loaded = False
            models_found = False
            loaded_models = []
            missing_models = []
            
            for timeframe in self.timeframes:
                # Her timeframe için en son kaydedilen modeli bul
                model_files = [f for f in os.listdir('saved_models') 
                             if f.startswith(f'lstm_{timeframe}') and f.endswith('.pth')]
                
                if not model_files:
                    print_warning(f"{timeframe} için model bulunamadı!")
                    missing_models.append(timeframe)
                    continue
                
                models_found = True
                
                # En son kaydedilen modeli al (tarih_saat bilgisine göre)
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join('saved_models', latest_model)
                
                try:
                    self.lstm_models[timeframe] = LSTMPredictor.load_model(model_path)
                    models_loaded = True
                    loaded_models.append(timeframe)
                    logger.info(f"{timeframe} modeli başarıyla yüklendi: {latest_model}")
                    print_success(f"{timeframe} modeli başarıyla yüklendi")
                except Exception as e:
                    logger.error(f"{timeframe} modeli yüklenirken hata: {str(e)}")
                    print_error(f"{timeframe} modeli yüklenirken hata: {str(e)}")
                    missing_models.append(timeframe)

            # Sonuçları göster
            print_section("MODEL DURUMU")
            
            if not models_found:
                print_error("Hiçbir model dosyası bulunamadı!")
                print_info("Bot çalışmak için eğitilmiş modellere ihtiyaç duyar.")
                print_info("Lütfen modelleri eğitin veya önceden eğitilmiş modelleri 'saved_models' klasörüne ekleyin.")
                return False
                
            if not models_loaded:
                print_error("Hiçbir model yüklenemedi!")
                print_info("Mevcut model dosyaları bozuk veya uyumsuz olabilir.")
                print_info("Lütfen modelleri yeniden eğitin.")
                return False
                
            if loaded_models and missing_models:
                print_warning("Bazı modeller eksik veya yüklenemedi!")
                print_info(f"✓ Yüklenen modeller: {', '.join(loaded_models)}")
                print_info(f"✗ Eksik modeller: {', '.join(missing_models)}")
                print_info("Bot kısmi işlevsellikle çalışabilir, ancak tam performans için tüm modellerin yüklenmesi önerilir.")
                user_input = input("\nEksik modelleri eğitmek ister misiniz? (y/n): ").strip().lower()
                if user_input == 'y':
                    print_section("EKSİK MODELLERİ EĞİTME")
                    print_info("Eksik modeller eğitilecek...")
                    
                    # Sadece eksik modelleri eğit
                    all_success = True
                    for timeframe in missing_models:
                        print_info(f"{timeframe} modeli eğitiliyor...")
                        try:
                            success = self.train_lstm_model(timeframe)
                            if success:
                                print_success(f"{timeframe} modeli başarıyla eğitildi!")
                                loaded_models.append(timeframe)
                                missing_models.remove(timeframe)
                            else:
                                print_error(f"{timeframe} modeli eğitimi başarısız oldu!")
                                all_success = False
                        except Exception as e:
                            logger.error(f"{timeframe} modeli eğitilirken hata: {str(e)}")
                            print_error(f"{timeframe} modeli eğitilirken hata: {str(e)}")
                            all_success = False
                    
                    # Eğitim sonuçlarını göster
                    print_section("EĞİTİM SONUÇLARI")
                    if all_success:
                        print_success("Tüm eksik modeller başarıyla eğitildi!")
                    else:
                        if missing_models:
                            print_warning("Bazı modeller hala eksik!")
                            print_info(f"✓ Mevcut modeller: {', '.join(loaded_models)}")
                            print_info(f"✗ Eksik modeller: {', '.join(missing_models)}")
                            print_info("Bot kısmi işlevsellikle çalışacak.")
                        else:
                            print_success("Tüm modeller başarıyla eğitildi!")
                    
                    return len(loaded_models) > 0  # En az bir model varsa True döndür
                else:
                    print_info("Bot mevcut modellerle çalışacak.")
                    return True
                
            print_success("Tüm modeller başarıyla yüklendi!")
            return True

        except Exception as e:
            logger.error(f"Model yükleme/oluşturma hatası: {str(e)}")
            print_error(f"Model yükleme hatası: {str(e)}")
            return False

    def train_models(self):
        """Modelleri eğit"""
        print_section("MODEL EĞİTİMİ")
        print("Eğitim seçenekleri:")
        print("1) Yerel (CPU)")
        print("2) Yerel (GPU)")
        
        choice = input("\nSeçiminiz (1/2): ").strip()
        
        # Model seçimi
        print_section("MODEL SEÇİMİ")
        print("1) Sadece LSTM")
        print("2) Sadece RL")
        print("3) LSTM + RL (Önerilen)")
        
        model_choice = input("\nSeçiminiz (1/2/3): ").strip()
        
        if model_choice not in ['1', '2', '3']:
            print_error("Geçersiz model seçimi!")
            return False

        if choice == '1':
            # CPU eğitimi
            device = torch.device('cpu')
        else:
            # GPU eğitimi
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print_success("GPU kullanılacak")
            else:
                print_warning("GPU bulunamadı, CPU kullanılacak")
                device = torch.device('cpu')

        # Eğitim verilerini hazırla
        print_info("Eğitim verileri hazırlanıyor...")
        train_data = self.data_processor.get_training_data()
        
        # Veri kontrolü
        if train_data is None:
            print_error("Eğitim verisi alınamadı!")
            return False
            
        # Veri kalitesi kontrolü
        if len(train_data) == 0:
            print_error("Eğitim verisi boş!")
            return False
            
        # Eksik veri kontrolü
        if train_data.isnull().values.any():
            missing_count = train_data.isnull().sum().sum()
            print_warning(f"{missing_count} eksik veri bulundu. Otomatik doldurma yapılıyor...")
            train_data = train_data.fillna(method='ffill').fillna(method='bfill')

        # LSTM modellerini eğit
        if model_choice in ['1', '3']:
            print_section("LSTM MODELLERİ EĞİTİMİ")
            
            # LSTM eğitimini başlat
            print_info("LSTM eğitimi başlatılıyor...")
            
            # Eğitim verilerini hazırla
            sequences = self.data_processor.prepare_sequences(
                train_data,
                MODEL_CONFIG['training']['sequence_length']
            )
            
            # LSTM modelini oluştur ve eğit
            lstm_model = LSTMPredictor(config=MODEL_CONFIG['lstm'])
            lstm_model.to(device)
            
            try:
                lstm_model.train_model(
                    sequences,
                    train_data['target'].values,
                    epochs=MODEL_CONFIG['training']['epochs'],
                    batch_size=MODEL_CONFIG['training']['batch_size'],
                    learning_rate=MODEL_CONFIG['training']['learning_rate']
                )
                print_success("LSTM eğitimi tamamlandı!")
            except Exception as e:
                print_error(f"LSTM eğitimi sırasında hata: {str(e)}")
                return False
        
        # RL modelini eğit
        if model_choice in ['2', '3']:
            print_section("RL MODEL EĞİTİMİ")
            
            # LSTM modelinin varlığını kontrol et
            lstm_model = self.lstm_models.get('5m')
            if lstm_model is None:
                print_error("RL için gerekli LSTM modeli bulunamadı!")
                return False
            
            # LSTM tahminlerini ekle
            print_info("LSTM tahminleri ekleniyor...")
            sequences = self.data_processor.prepare_sequences(
                train_data,
                MODEL_CONFIG['training']['sequence_length']
            )
            with torch.no_grad():
                lstm_predictions = lstm_model(sequences).numpy()
            train_data['lstm_prediction'] = lstm_predictions
            
            # RL modelini oluştur ve eğit
            try:
                env_params = {
                    'df': train_data,
                    'window_size': MODEL_CONFIG['rl']['window_size'],
                    'initial_balance': self.risk_manager.initial_balance if self.risk_manager else 10000.0,
                    'commission': TRADING_CONFIG['commission']
                }
                
                self.rl_trader = RLTrader(lstm_model=lstm_model, env_params=env_params)
                self.rl_trader.train(
                    total_timesteps=MODEL_CONFIG['rl']['total_timesteps'],
                    log_interval=MODEL_CONFIG['rl']['log_interval']
                )
                print_success("RL eğitimi tamamlandı!")
            except Exception as e:
                print_error(f"RL eğitimi sırasında hata: {str(e)}")
                return False

        print_section("MODEL EĞİTİMİ TAMAMLANDI")
        print_success("Tüm modeller başarıyla eğitildi!")
        return True

    def execute_trade(self, timeframe, lstm_pred, rl_action):
        """Execute trades based on model predictions"""
        try:
            print_section(f"TİCARET YÜRÜTME BAŞLANGIÇ ({timeframe})")
            
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
            symbol_info = self.mt5.symbol_info("XAUUSD")
            if symbol_info is None:
                logger.error("Sembol bilgisi alınamadı")
                return
            
            # Stop loss ve take profit hesaplama için risk yöneticisini kullan
            if trade_type == mt5.ORDER_TYPE_BUY:
                position_type = "BUY"
                entry_price = symbol_info.ask
            else:  # SELL
                position_type = "SELL"
                entry_price = symbol_info.bid
            
            # Stop loss hesaplama
            sl_price = self.risk_manager.calculate_stop_loss(entry_price, atr, position_type)
            
            # Take profit hesaplama
            tp_price = self.risk_manager.calculate_take_profit(entry_price, sl_price, TRADING_CONFIG['risk_reward_ratio'])
            
            # Lot büyüklüğü hesaplama
            lot_size = self.risk_manager.calculate_position_size(entry_price, sl_price)
            
            # Lot büyüklüğü sınırlamaları
            lot_size = max(min(lot_size, symbol_info.volume_max), symbol_info.volume_min)
            
            print(f"Stop Loss: {sl_price:.2f} (mesafe: {abs(entry_price - sl_price):.2f})")
            print(f"Take Profit: {tp_price:.2f} (mesafe: {abs(entry_price - tp_price):.2f})")
            print(f"Hesaplanan lot: {lot_size}")
            
            # İşlemi position manager ile aç
            ticket = self.position_manager.open_position(
                symbol="XAUUSD",
                order_type=trade_type,
                volume=lot_size,
                price=entry_price,
                sl=sl_price,
                tp=tp_price,
                comment=f"Auto Trade {timeframe}"
            )
            
            if ticket:
                print(f"İşlem başarıyla gerçekleşti! Ticket: {ticket}")
                self.risk_manager.update_balance(0)  # İşlem başlangıçta sıfır kar/zarar ile başlar
            else:
                error_msg = f"İşlem başarısız"
                logger.error(error_msg)
                print(error_msg)
            
        except Exception as e:
            logger.error(f"İşlem hatası: {str(e)}")
            print(f"İşlem hatası: {str(e)}")
        finally:
            print_section("TİCARET YÜRÜTME BİTİŞ")

    def stop(self):
        """Stop the trading bot gracefully"""
        try:
            print_section("BOT DURDURULUYOR")
            
            # Handle training interruption first if training is in progress
            if self.training_in_progress:
                print_warning("Eğitim devam ederken bot durdurma talebi alındı. Eğitim güvenli bir şekilde durduruluyor...")
                
                # Signal the training to stop
                self.training_interrupted = True
                self.training_stop_event.set()
                
                # Wait for training to finish gracefully (with timeout)
                if self.training_thread and self.training_thread.is_alive():
                    print_info("Eğitim durana kadar bekleniyor...")
                    self.training_thread.join(timeout=10)  # Wait up to 10 seconds
                    
                    # If training is still running after timeout, take more aggressive measures
                    if self.training_thread.is_alive():
                        print_warning("Eğitim işlemi zaman aşımına uğradı, zorla durduruluyor...")
                        # We can't directly terminate the thread, but we've set the flag for the training loop to check
                
                print_success("Eğitim güvenli bir şekilde durduruldu")
                self.training_in_progress = False
            
            # Stop system monitor first
            if self.system_monitor:
                self.system_monitor.stop_monitoring()
                print_info("Sistem monitörü durduruldu")
            
            # Close MT5 connection
            if self.mt5 and self.mt5.connected:
                self.mt5.disconnect()
                print_info("MT5 bağlantısı güvenli bir şekilde kapatıldı.")
                print_info("MT5 bağlantısı kapatıldı")
            
            # Close any open positions
            if self.position_manager:
                open_positions = self.position_manager.get_open_positions()
                if open_positions:
                    print_warning(f"{len(open_positions)} açık pozisyon bulundu, kapatılıyor...")
                    for pos in open_positions:
                        self.position_manager.close_position(pos)
                    print_success("Tüm pozisyonlar kapatıldı")
            
            print_success("Bot başarıyla durduruldu!")
            return True
            
        except Exception as e:
            print_error(f"Bot durdurma hatası: {str(e)}")
            logger.error(f"Bot durdurma hatası: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def shutdown(self):
        """Bot'u güvenli bir şekilde kapat ve kaynakları temizle"""
        try:
            logger.info("\n==================================================")
            logger.info("BOT KAPATILIYOR")
            logger.info("==================================================\n")
            
            # Önce bot'u durdur
            self.stop()
            
            # Belleği temizle
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Log handler'ları kapat
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            return True
            
        except Exception as e:
            print(f"\nKapatma hatası: {str(e)}")
            return False

    def execute_trades(self, timeframe):
        """
        Belirli bir zaman dilimi için ticaret sinyallerini üret ve işlemleri yürüt
        
        Args:
            timeframe (str): İşlem yapılacak zaman dilimi (örn. '5m', '15m', '1h')
        """
        try:
            # Veri toplama
            candles = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=100)
            if candles is None or len(candles) < 100:
                logger.error(f"{timeframe} için yeterli veri yok")
                return
            
            # Teknik göstergeleri hesapla
            df = self.data_processor.add_technical_indicators(candles)
            if df is None:
                logger.error("Teknik göstergeler hesaplanamadı")
                return
            
            # LSTM tahmini al
            lstm_model = self.lstm_models.get(timeframe)
            if lstm_model is None:
                logger.error(f"{timeframe} için LSTM modeli bulunamadı")
                return
            
            # Veriyi modele uygun formata dönüştür
            lstm_data = self.data_processor.prepare_model_data(df)
            if lstm_data is None:
                logger.error("Model verisi hazırlanamadı")
                return
            
            # LSTM tahmini
            with torch.no_grad():
                lstm_pred = lstm_model(lstm_data)
                
            # RL tahmini (eğer model varsa)
            rl_state = self.data_processor.prepare_rl_state(df)
            rl_action = None
            if self.rl_trader is not None:
                rl_action = self.rl_trader.predict(rl_state)
            
            # İşlemi yürüt
            self.execute_trade(timeframe, lstm_pred, rl_action)
            
        except Exception as e:
            logger.error(f"{timeframe} için işlem hatası: {str(e)}")
            print_error(f"{timeframe} için işlem hatası: {str(e)}")

    def start(self):
        """Start the trading bot"""
        try:
            # Welcome message
            print_section(
                "ALTIN TİCARET BOTUNA HOŞGELDİNİZ",
                "WELCOME TO GOLD TRADING BOT"
            )
            print_info(
                "Bot başlatılıyor...",
                "Bot is starting..."
            )

            # MT5 Connection
            print_section(
                "MT5 BAĞLANTISI",
                "MT5 CONNECTION"
            )
            print_info(
                "MetaTrader 5'e bağlanılıyor...",
                "Connecting to MetaTrader 5..."
            )
            
            self.mt5 = MT5Connector()
            if not self.mt5.connect():
                print_error("MT5 bağlantısı başarısız!")
                return False
            
            # Hesap ve bağlantı bilgileri
            print_success(f"MT5 bağlantısı başarılı! ({self.mt5.account_info.server})")
            print_balance_info(f"Hesap: {self.mt5.account_info.login} | Bakiye: ${self.mt5.account_info.balance:,.2f}")
            print_trade_info(f"XAUUSD Spread: {self.mt5.symbol_info('XAUUSD').spread} puan | Lot: {self.mt5.symbol_info('XAUUSD').volume_min:.2f} - {self.mt5.symbol_info('XAUUSD').volume_max:.1f}")

            # Initialize components
            print_info("Sistem bileşenleri hazırlanıyor...")
            self.data_processor = DataProcessor()
            self.risk_manager = RiskManager(self.mt5, TRADING_CONFIG['risk_per_trade'], TRADING_CONFIG['max_daily_loss'])
            self.system_monitor = SystemMonitor()

            # Initial Tests
            print_section("SİSTEM KONTROLÜ")
            response = input("Sistem kontrolü yapmak ister misiniz? (y/n): ")
            if response.lower() == 'y':
                print_info("Sistem kontrolleri başlatılıyor...")
                
                # MT5 connection test
                if not self.mt5.test_connection():
                    print_error("MT5 bağlantı testi başarısız!")
                    return False
                
                # Data retrieval test
                if not self.test_data_retrieval():
                    print_error("Veri alımı testi başarısız!")
                    return False
                
                # System resources test
                memory_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                if memory_available < 1.5:  # 1.5GB minimum gereksinim
                    print_warning(f"Kritik düşük bellek! ({memory_available:.1f}GB)")
                    print_info("Önerilen minimum bellek: 2GB")
                    print_info("Performans sorunları yaşanabilir veya eğitim başarısız olabilir.")
                    user_continue = input("Devam etmek istiyor musunuz? (y/n): ").strip().lower()
                    if user_continue != 'y':
                        return False
                elif memory_available < 2.0:  # 2GB önerilen minimum
                    print_warning(f"Düşük bellek: {memory_available:.1f}GB")
                    print_info("Önerilen: 2GB veya üzeri")
                else:
                    print_success(f"Bellek testi başarılı! ({memory_available:.1f}GB kullanılabilir)")

            # Model Status
            print_section("MODEL DURUMU")
            retrain_input = input("Modelleri eğitmek ister misiniz? (y/n): ").strip().lower()
            
            if retrain_input == 'y':
                print_info("Model eğitimi başlatılıyor...")
                if not self.train_models():
                    print_error("Model eğitimi başarısız! Bot çalışamaz.")
                    return False
            else:
                # Mevcut modelleri yüklemeyi dene
                print_section("MODEL YÜKLEME")
                print_info("Mevcut modeller kontrol ediliyor...")
                if not self.load_or_create_models():
                    print_error("Model yükleme başarısız!")
                    print_info("Bot çalışmak için eğitilmiş modellere ihtiyaç duyar.")
                    
                    while True:
                        print("\nSeçenekler:")
                        print("1) Modelleri eğit")
                        print("2) Çıkış yap")
                        choice = input("\nSeçiminiz (1/2): ").strip()
                        
                        if choice == "1":
                            print_info("Model eğitimi başlatılıyor...")
                            if not self.train_models():
                                print_error("Model eğitimi başarısız! Bot çalışamaz.")
                                return False
                            break
                        elif choice == "2":
                            print_info("Bot kapatılıyor...")
                            return False
                        else:
                            print_warning("Geçersiz seçim! Lütfen 1 veya 2 girin.")
            
            print_section("BOT HAZIR")
            print_success("Tüm sistemler hazır!")
            
            # Sistem monitörünü başlat
            if hasattr(self, 'system_monitor') and self.system_monitor:
                self.system_monitor.start_monitoring()
                print_success("Sistem monitörü başlatıldı!")
            
            # Bot'u çalıştır
            self.run()
            
        except Exception as e:
            print_error(f"Bot başlatma hatası: {str(e)}")
            logger.error(f"Bot başlatma hatası: {str(e)}")
            return False

    def train_lstm_model(self, timeframe):
        """Train LSTM model"""
        try:
            print_section(
                f"{timeframe} MODELİ EĞİTİMİ",
                f"TRAINING {timeframe} MODEL"
            )
            print_info(
                f"{timeframe} zaman dilimi için LSTM modeli eğitiliyor...",
                f"Training LSTM model for {timeframe} timeframe..."
            )
            print_warning(
                "Bu işlem birkaç saat sürebilir, lütfen bekleyin...",
                "This process may take several hours, please wait..."
            )
            
            # Set the training state flags
            self.training_in_progress = True
            self.current_training_model = f"LSTM_{timeframe}"
            self.training_interrupted = False
            self.training_stop_event.clear()
            
            # Create checkpoint directory if it doesn't exist
            checkpoint_dir = os.path.join(self.base_dir, 'saved_models', 'checkpoints')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            
            # Set checkpoint path
            self.checkpoint_path = os.path.join(checkpoint_dir, f"lstm_{timeframe}_checkpoint.pth")
            
            # Eğitim verilerini hazırla
            print_info("Eğitim verileri hazırlanıyor...")
            train_data = self.data_processor.get_training_data(timeframe)
            if train_data is None:
                raise Exception(f"{timeframe} için veri alınamadı")
            
            # Minimum veri kontrolü
            if len(train_data) < 1000:  # Minimum veri gereksinimi
                raise Exception(f"{timeframe} için yeterli veri yok: {len(train_data)} < 1000")
            
            # Veri kalitesi kontrolü
            if train_data.isnull().values.any():
                missing_count = train_data.isnull().sum().sum()
                print_warning(f"{missing_count} eksik veri bulundu. Otomatik doldurma yapılıyor...")
                train_data = train_data.fillna(method='ffill').fillna(method='bfill')
            
            # Model parametrelerini al
            params = MODEL_CONFIG['lstm']
            sequence_length = params.get('sequence_length', 60)
            hidden_size = params.get('hidden_size', 64)
            num_layers = params.get('num_layers', 2)
            dropout = params.get('dropout', 0.2)
            learning_rate = params.get('learning_rate', 0.001)
            batch_size = params.get('batch_size', 32)
            epochs = params.get('epochs', 50)
            validation_split = MODEL_CONFIG['training'].get('validation_split', 0.1)
            
            # Veriyi model için hazırla
            print_info("Eğitim ve doğrulama sekansları hazırlanıyor...")
            X, y = self.data_processor.prepare_sequences(train_data, sequence_length)
            if X is None or y is None:
                raise Exception("Veri hazırlama başarısız!")
            
            # Veri boyutlarını kontrol et
            if len(X) < batch_size:
                raise Exception(f"Yetersiz veri: {len(X)} örnek < {batch_size} batch size")
            
            # Veriyi eğitim ve doğrulama setlerine böl
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print_info(f"Veri bölündü: {len(X_train)} eğitim, {len(X_val)} doğrulama örneği")
            
            # Model oluştur
            input_size = X.shape[2]  # Özellik sayısı
            model = LSTMPredictor(config={
                'lstm': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'bidirectional': True
                },
                'batch_norm': {
                    'momentum': 0.1,
                    'eps': 1e-5
                },
                'attention': {
                    'dims': [hidden_size * 2, 64, 1],
                    'dropout': 0.2
                }
            })
            
            # Check if checkpoint exists
            if os.path.exists(self.checkpoint_path):
                print_info(f"Eğitim kontrol noktası bulundu: {self.checkpoint_path}")
                user_input = input("Eğitimi kontrol noktasından devam ettirmek ister misiniz? (y/n): ")
                if user_input.lower() == 'y':
                    try:
                        model.load_checkpoint(self.checkpoint_path)
                        print_success("Kontrol noktası başarıyla yüklendi!")
                    except Exception as e:
                        print_error(f"Kontrol noktası yükleme hatası: {str(e)}")
                        print_info("Yeni eğitim başlatılıyor...")
            
            # Eğitim başlangıç zamanı
            start_time = time.time()
            
            # Bellek temizliği
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create a wrapper for the training function that can be run in a thread
            def training_thread_function():
                try:
                    # Modeli eğit
                    print_info(f"Model eğitimi başlıyor... (Epochs: {epochs}, Batch Size: {batch_size})")
                    
                    # Add interrupt_check function to be passed to the model
                    def interrupt_check():
                        return self.training_interrupted or self.training_stop_event.is_set()
                    
                    # Add checkpoint_save function to be passed to the model
                    def checkpoint_save(model_state):
                        try:
                            # Save current model state as checkpoint
                            torch.save(model_state, self.checkpoint_path)
                            logger.info(f"Kontrol noktası kaydedildi: {self.checkpoint_path}")
                        except Exception as e:
                            logger.error(f"Kontrol noktası kaydetme hatası: {str(e)}")
                    
                    # Train with interrupt and checkpoint handlers
                    history = model.train_model(
                        train_sequences=X_train,
                        train_targets=y_train,
                        val_sequences=X_val,
                        val_targets=y_val,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        verbose=True,
                        interrupt_check=interrupt_check,
                        checkpoint_save=checkpoint_save,
                        checkpoint_interval=5  # Save checkpoint every 5 epochs
                    )
                    
                    # Eğitim süresini hesapla
                    training_time = time.time() - start_time
                    hours = int(training_time // 3600)
                    minutes = int((training_time % 3600) // 60)
                    
                    # Check if training was interrupted
                    if interrupt_check():
                        print_warning("Eğitim kullanıcı tarafından durduruldu.")
                        # Save one final checkpoint if interrupted
                        checkpoint_save(model.state_dict())
                    else:
                        # Modeli kaydet
                        save_path = f"saved_models/lstm_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                        try:
                            model.save_checkpoint(save_path)
                            print_success(f"Model kaydedildi: {save_path}")
                        except Exception as e:
                            print_error(f"Model kaydetme hatası: {str(e)}")
                    
                    # Modeli sözlüğe ekle
                    self.lstm_models[timeframe] = model
                    
                    # Performans metriklerini kaydet
                    try:
                        metrics_path = f"saved_models/lstm_{timeframe}_metrics.json"
                        metrics = {
                            'training_time': training_time,
                            'epochs': epochs,
                            'batch_size': batch_size,
                            'data_size': len(X),
                            'train_size': len(X_train),
                            'val_size': len(X_val),
                            'interrupted': interrupt_check(),
                            'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        
                        # Add history data if available
                        if history and 'train_losses' in history:
                            metrics['train_losses'] = history['train_losses']
                        if history and 'val_losses' in history:
                            metrics['val_losses'] = history['val_losses']
                        
                        with open(metrics_path, 'w') as f:
                            json.dump(metrics, f, indent=2)
                        
                        print_info(f"Performans metrikleri kaydedildi: {metrics_path}")
                    except Exception as e:
                        print_error(f"Metrik kaydetme hatası: {str(e)}")
                    
                    # Clear training state flags
                    self.training_in_progress = False
                    self.current_training_model = None
                    
                    return model
                    
                except Exception as e:
                    print_error(f"Eğitim thread hatası: {str(e)}")
                    logger.error(f"Eğitim thread hatası: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Clear training state flags
                    self.training_in_progress = False
                    self.current_training_model = None
                    return None
            
            # Create and start the training thread
            self.training_thread = threading.Thread(target=training_thread_function)
            self.training_thread.daemon = True  # Make thread daemonic so it won't block program exit
            self.training_thread.start()
            
            # Wait for training to complete
            while self.training_thread.is_alive():
                try:
                    # Check for interruption by user (e.g., KeyboardInterrupt)
                    self.training_thread.join(1.0)  # Check every second
                    
                    # Show a heartbeat message every minute
                    elapsed_time = time.time() - start_time
                    if int(elapsed_time) % 60 == 0:
                        mins = int(elapsed_time // 60)
                        hrs = mins // 60
                        mins = mins % 60
                        print_info(f"Eğitim devam ediyor... Geçen süre: {hrs:02d}:{mins:02d}")
                        
                except KeyboardInterrupt:
                    # Handle keyboard interrupt (Ctrl+C)
                    print_warning("\nKullanıcı tarafından eğitim durdurma talebi alındı (Ctrl+C)")
                    self.training_interrupted = True
                    self.training_stop_event.set()
                    print_info("Eğitim durana kadar bekleyin...")
                    
                    # Wait for training thread to finish with timeout
                    self.training_thread.join(timeout=30)
                    if self.training_thread.is_alive():
                        print_warning("Eğitim işlemi zaman aşımına uğradı!")
                    break
            
            # Check if training completed successfully
            if self.training_interrupted:
                print_warning("Eğitim kesintiye uğradı. Kontrol noktasından daha sonra devam edebilirsiniz.")
                return None
            else:
                print_success("Eğitim başarıyla tamamlandı!")
                return self.lstm_models.get(timeframe)
            
        except Exception as e:
            print_error(f"Model eğitimi başarısız: {str(e)}")
            logger.error(f"Model eğitimi başarısız: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Clear training state flags
            self.training_in_progress = False
            self.current_training_model = None
            return None

def signal_handler(signum, frame):
    """Handle termination signals"""
    print_section("\nSİNYAL ALINDI: CTRL+C")
    print_info("Bot güvenli bir şekilde durduruluyor...")
    
    try:
        if bot:
            # If bot is in training, set the interruption flag
            if hasattr(bot, 'training_in_progress') and bot.training_in_progress:
                print_warning(f"Eğitim süreci devam ediyor: {bot.current_training_model}")
                print_info("Eğitim güvenli bir şekilde durdurulacak...")
                
                # Set the training interrupted flag to true
                bot.training_interrupted = True
                # Signal the training thread to stop
                bot.training_stop_event.set()
                
                # Give some time for the training to save checkpoint
                print_info("Eğitim durduruluyor, lütfen bekleyin...")
                time.sleep(3)
            
            # Now stop the bot
            if bot.stop():
                print_success("Bot güvenli bir şekilde durduruldu")
            else:
                print_error("Bot durdurma başarısız!")
        else:
            print_error("Bot bulunamadı!")
    except Exception as e:
        print_error(f"Sinyal işleme hatası: {str(e)}")
        logger.error(f"Sinyal işleme hatası: {str(e)}")
        logger.error(traceback.format_exc())
    
    sys.exit(0)

def main():
    """Ana program"""
    global bot
    bot = None
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print_section("ALTIN TİCARET BOTUNA HOŞGELDİNİZ")
        print_info("Bot başlatılıyor...")
        
        # Create bot instance
        bot = XAUUSDTradingBot()
        
        # Initialize bot
        if not bot.initialize():
            print_error("Bot başlatılamadı!")
            return
            
        # Bot'u çalıştır
        bot.run()
        
    except KeyboardInterrupt:
        print_section("\nKULLANICI KESİNTİSİ")
        if bot:
            bot.stop()
        sys.exit(0)
    except Exception as e:
        print_error(f"Beklenmeyen hata: {str(e)}")
        logger.error(f"Beklenmeyen hata: {str(e)}")
        logger.error(traceback.format_exc())
        if bot:
            bot.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 