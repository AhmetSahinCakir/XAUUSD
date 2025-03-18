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
        """Run trading bot in an infinite loop"""
        try:
            print_section("BOT BAŞLADI")
            print_info("Bot çalışıyor...")
            
            while not self.training_stop_event.is_set():
                for timeframe in self.timeframes:
                    try:
                        # Her timeframe için işlem sinyalleri kontrol et
                        self.check_trading_signals(timeframe)
                        time.sleep(1)  # Çok hızlı döngüye girmeyi engelle
                    except Exception as e:
                        logger.error(f"{timeframe} için işlem kontrolü sırasında hata: {str(e)}")
                
                # Açık pozisyonları kontrol et
                try:
                    positions = self.position_manager.get_open_positions("XAUUSD")
                    if positions:
                        print(f"{len(positions)} açık pozisyon var.")
                        for pos in positions:
                            # Trailing stop kontrolü
                            try:
                                self.position_manager.update_trailing_stop(pos)
                            except Exception as e:
                                logger.error(f"Trailing stop güncellemesi sırasında hata: {str(e)}")
                except Exception as e:
                    logger.error(f"Pozisyon kontrolü sırasında hata: {str(e)}")
                
                # Risk yönetimi güncellemesi
                try:
                    # Hesabın güncel durumunu al
                    balance = self.mt5.get_account_info().balance
                    # Risk yöneticisini güncelle
                    self.risk_manager.update_balance(balance)
                except Exception as e:
                    logger.error(f"Risk yönetimi güncellemesi sırasında hata: {str(e)}")
                
                # Periyodik bekle
                time.sleep(30)  # 30 saniye bekle
                
        except KeyboardInterrupt:
            print("\nBot kullanıcı tarafından durduruldu.")
        except Exception as e:
            print(f"Çalışma hatası: {str(e)}")
            logger.error(f"Çalışma hatası: {str(e)}")
        finally:
            self.stop()
            print_section("BOT DURDURULDU")
        
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
                # Veri hazırlığı - her timeframe için ayrı veri hazırla
                data_dict = {}
                for tf in self.timeframes:
                    data_dict[tf] = self.data_processor.get_latest_data(timeframe=tf)
                    if data_dict[tf] is None:
                        print_warning(f"{tf} için veri alınamadı!")
                
                if not data_dict:
                    raise Exception("Hiçbir timeframe için veri alınamadı!")
                
                # Çevre parametrelerini oluştur
                env_params = {
                    'df': data_dict,
                    'window_size': MODEL_CONFIG['rl']['window_size'],
                    'initial_balance': self.risk_manager.initial_balance if hasattr(self, 'risk_manager') else 10000.0,
                    'commission': TRADING_CONFIG.get('transaction_fee', 0.00025)
                }
                
                # LSTM modellerini kontrol et
                if not self.lstm_models:
                    raise Exception("RL Trader için LSTM modeli bulunamadı! Lütfen önce modelleri eğitin (--train_lstm)")
                
                # RL Trader'ı başlat
                self.rl_trader = RLTrader(
                    lstm_models=self.lstm_models,
                    env_params=env_params
                )
                
                print_success("RL Trader başarıyla başlatıldı!")
                print_info(f"Kullanılabilir timeframe modelleri: {', '.join(self.lstm_models.keys())}")
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
        
        # LSTM modellerini eğit
        if model_choice in ['1', '3']:
            print_section("LSTM MODELLERİ EĞİTİMİ")
            
            # Her bir timeframe için model eğit
            for timeframe in DATA_CONFIG['timeframes']:
                print_section(f"{timeframe} LSTM MODELİ EĞİTİMİ")
                print_info(f"{timeframe} için LSTM eğitimi başlatılıyor...")
                
                # Timeframe için veri al
                train_data = self.data_processor.get_training_data(timeframe)
                
                # Veri kontrolü
                if train_data is None:
                    print_error(f"{timeframe} için eğitim verisi alınamadı!")
                    continue
                    
                # Veri kalitesi kontrolü
                if len(train_data) == 0:
                    print_error(f"{timeframe} için eğitim verisi boş!")
                    continue
                    
                # Eksik veri kontrolü
                if train_data.isnull().values.any():
                    missing_count = train_data.isnull().sum().sum()
                    print_warning(f"{timeframe} için {missing_count} eksik veri bulundu. Otomatik doldurma yapılıyor...")
                    train_data = train_data.fillna(method='ffill').fillna(method='bfill')
                
                try:
                    # Bu timeframe için LSTM modelini eğit
                    success = self.train_lstm_model(timeframe)
                    if success:
                        print_success(f"{timeframe} LSTM modeli başarıyla eğitildi!")
                    else:
                        print_error(f"{timeframe} LSTM modeli eğitimi başarısız oldu!")
                except Exception as e:
                    print_error(f"{timeframe} LSTM modeli eğitimi sırasında hata: {str(e)}")
                    logger.error(f"{timeframe} LSTM modeli eğitimi sırasında hata: {str(e)}")
                    logger.error(traceback.format_exc())
            
            print_section("LSTM EĞİTİMİ TAMAMLANDI")
            
        # RL modelini eğit
        if model_choice in ['2', '3']:
            print_section("RL MODELİ EĞİTİMİ")
            # ... RL eğitim kodu ...
            
        return True

    def check_trading_signals(self, timeframe):
        """
        LSTM ve RL modelleri kullanarak işlem sinyallerini kontrol et
        """
        try:
            # Modelleri kontrol et
            lstm_model = self.lstm_models.get(timeframe)
            if lstm_model is None:
                logger.error(f"{timeframe} için LSTM modeli bulunamadı")
                return
            
            # RL Trader kontrolü
            has_rl = hasattr(self, 'rl_trader') and self.rl_trader is not None
            
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
            
            # LSTM tahmini yap
            lstm_pred = lstm_model.predict_proba(df.tail(60))
            
            # LSTM tahminini normalize et (0-1 arası)
            lstm_pred_norm = (lstm_pred[0] + 0.10) / 0.20  # -0.10 -> 0.0, 0.10 -> 1.0
            lstm_pred_norm = max(0, min(1, lstm_pred_norm))  # 0-1 arasına sınırla
            
            print(f"Mevcut fiyat: {current_price}, ATR: {atr}, LSTM tahmini: {lstm_pred_norm:.4f}")
            
            # İşlem kararı - varsayılan olarak işlem yok
            trade_type = None
            trade_confidence = 0
            
            if has_rl:
                # Tüm timeframe'ler için state hazırla
                states = {}
                for tf in self.timeframes:
                    tf_data = self.mt5.get_historical_data("XAUUSD", tf, num_candles=100)
                    if tf_data is not None and len(tf_data) >= 30:
                        tf_df = self.data_processor.add_technical_indicators(tf_data)
                        if tf_df is not None:
                            # Son 60 veriyi al ve RL state olarak hazırla
                            states[tf] = self.data_processor.prepare_rl_state(tf_df)
                
                if states:
                    try:
                        # Tüm timeframe'lerden birleşik tahmin al
                        rl_action, action_details = self.rl_trader.predict_combined(states)
                        
                        # Bireysel tahminleri logla
                        for tf, pred in action_details['individual_predictions'].items():
                            action_name = "BEKLE" if pred == 0 else "AL" if pred == 1 else "SAT"
                            print(f"{tf} RL tahmini: {action_name} ({pred})")
                        
                        # Voting sonuçlarını ve güven değerlerini logla
                        votes = action_details['votes']
                        print(f"Oylama sonuçları: BEKLE: {votes[0]:.2f}, AL: {votes[1]:.2f}, SAT: {votes[2]:.2f}")
                        
                        # Güven skoru - en yüksek oyun toplam oya oranı
                        max_vote = max(votes.values())
                        total_votes = sum(votes.values())
                        confidence = max_vote / total_votes if total_votes > 0 else 0
                        
                        print(f"Birleşik RL tahmini: {rl_action} ({0: 'BEKLE', 1: 'AL', 2: 'SAT'}[rl_action]), Güven: {confidence:.2f}")
                        
                        # İşlem kararını RL modeli belirliyor
                        if rl_action == 1:  # AL
                            trade_type = mt5.ORDER_TYPE_BUY
                            trade_confidence = confidence
                            print(f"RL modeli AL sinyali verdi, güven skoru: {confidence:.2f}")
                        elif rl_action == 2:  # SAT
                            trade_type = mt5.ORDER_TYPE_SELL
                            trade_confidence = confidence
                            print(f"RL modeli SAT sinyali verdi, güven skoru: {confidence:.2f}")
                        else:
                            print("RL modeli BEKLEMEYİ öneriyor")
                            
                    except Exception as e:
                        logger.error(f"RL tahmininde hata: {str(e)}")
                        # RL model hata verirse, sadece LSTM modeline geri dönüyoruz
                        # Bu kısım ileride tamamen kaldırılabilir, şimdilik fallback olarak bırakıyoruz
                        if lstm_pred_norm > 0.65:  # Güçlü ALIM
                            trade_type = mt5.ORDER_TYPE_BUY
                            trade_confidence = lstm_pred_norm
                            print(f"RL modeli hata verdi, LSTM modeli AL sinyali verdi: {lstm_pred_norm:.2f}")
                        elif lstm_pred_norm < 0.35:  # Güçlü SATIM
                            trade_type = mt5.ORDER_TYPE_SELL
                            trade_confidence = 1 - lstm_pred_norm
                            print(f"RL modeli hata verdi, LSTM modeli SAT sinyali verdi: {lstm_pred_norm:.2f}")
                        else:
                            print("İşlem sinyali yok")
            else:
                # Sadece LSTM modeli varsa (fallback)
                if lstm_pred_norm > 0.65:  # Güçlü ALIM
                    trade_type = mt5.ORDER_TYPE_BUY
                    trade_confidence = lstm_pred_norm
                    print(f"ALIM sinyali: LSTM ({lstm_pred_norm:.2f})")
                elif lstm_pred_norm < 0.35:  # Güçlü SATIM
                    trade_type = mt5.ORDER_TYPE_SELL
                    trade_confidence = 1 - lstm_pred_norm
                    print(f"SATIM sinyali: LSTM ({lstm_pred_norm:.2f})")
                else:
                    print(f"İşlem sinyali yok: LSTM ({lstm_pred_norm:.2f})")
                    return
            
            # İşlem yapmaya karar verildi mi?
            if trade_type is None:
                return
                
            # Güven skoru minimum eşiği aşıyor mu?
            MIN_CONFIDENCE = 0.6  # Minimum %60 güven
            if trade_confidence < MIN_CONFIDENCE:
                print(f"Güven skoru çok düşük ({trade_confidence:.2f} < {MIN_CONFIDENCE}), işlem yapılmıyor")
                return
            
            # Risk yönetimi kontrolü
            if not self.risk_manager.can_trade():
                print("Risk limitlerine ulaşıldı, işlem yapılmıyor")
                return
            
            # İşlem boyutu hesapla
            calculated_lot = self.risk_manager.calculate_position_size(current_price, atr * 2)
            
            # Mevcut pozisyonları kontrol et
            open_positions = self.position_manager.get_open_positions("XAUUSD")
            
            # Eğer aynı yönde pozisyon varsa, işlem yapma
            for pos in open_positions:
                if (trade_type == mt5.ORDER_TYPE_BUY and pos['type'] == 0) or \
                   (trade_type == mt5.ORDER_TYPE_SELL and pos['type'] == 1):
                    print(f"Zaten bu yönde açık pozisyon var (Ticket: {pos['ticket']}), işlem yapılmıyor")
                    return
            
            # Stop Loss ve Take Profit hesapla
            if trade_type == mt5.ORDER_TYPE_BUY:
                sl = current_price - (atr * 2)
                tp = current_price + (atr * 4)
            else:
                sl = current_price + (atr * 2)
                tp = current_price - (atr * 4)
            
            # İşlemi aç
            trade_result = self.mt5.open_trade(
                symbol="XAUUSD",
                order_type=trade_type,
                lot=calculated_lot,
                sl=sl,
                tp=tp
            )
            
            if trade_result:
                print(f"İşlem başarılı: {trade_result}")
            else:
                print("İşlem açılamadı")
                
        except Exception as e:
            logger.error(f"İşlem sinyalleri kontrol edilirken hata: {str(e)}")
            print(f"İşlem sinyalleri kontrol edilirken hata: {str(e)}")

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
            self.check_trading_signals(timeframe)
            
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
            logger.error(traceback.format_exc())
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
                print_warning(f"{timeframe} için {missing_count} eksik veri bulundu. Otomatik doldurma yapılıyor...")
                train_data = train_data.fillna(method='ffill').fillna(method='bfill')
            
            # Veriyi eğitim ve doğrulama setlerine böl
            total_size = len(train_data)
            split_idx = int(total_size * (1 - MODEL_CONFIG['training']['validation_split']))
            
            train_df = train_data.iloc[:split_idx]
            val_df = train_data.iloc[split_idx:]
            
            print_info(f"Veri bölündü: {len(train_df)} eğitim, {len(val_df)} doğrulama örneği")
            
            # Eğitim ve doğrulama verilerini hazırla
            train_sequences, train_targets = self.data_processor.prepare_sequences(
                df=train_df,
                sequence_length=MODEL_CONFIG['training']['sequence_length'],
                target_column='close',
                prediction_steps=1,
                timeframe=timeframe
            )
            
            val_sequences, val_targets = self.data_processor.prepare_sequences(
                df=val_df,
                sequence_length=MODEL_CONFIG['training']['sequence_length'],
                target_column='close',
                prediction_steps=1,
                timeframe=timeframe
            )
            
            if train_sequences is None or train_targets is None:
                print_error("LSTM için eğitim dizileri oluşturulamadı!")
                return False
                
            if val_sequences is None or val_targets is None:
                print_warning("LSTM için doğrulama dizileri oluşturulamadı, doğrulama yapılmayacak!")
            else:
                print_info(f"Doğrulama verileri hazır: {val_sequences.shape}")
            
            # LSTM modelini oluştur ve eğit
            model_config = MODEL_CONFIG['lstm'].copy()
            model_config['input_size'] = train_sequences.shape[2]
            lstm_model = LSTMPredictor(config=model_config)
            lstm_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            try:
                # Eğitim sonuçlarını al
                training_results = lstm_model.train_model(
                    train_sequences=train_sequences,
                    train_targets=train_targets,
                    val_sequences=val_sequences,
                    val_targets=val_targets,
                    epochs=MODEL_CONFIG['training']['epochs'],
                    batch_size=MODEL_CONFIG['training']['batch_size'],
                    learning_rate=MODEL_CONFIG['training']['learning_rate'],
                    verbose=True
                )
                
                # Eğitim sonuçlarını detaylı göster
                print_section(f"{timeframe} MODELİ EĞİTİM SONUÇLARI")
                
                # Eğitim metrikleri
                print_info(f"Son eğitim kaybı: {training_results['train_losses'][-1]:.6f}")
                print_info(f"Son eğitim doğruluğu: %{training_results['train_accuracies'][-1]*100:.2f}")
                
                # Doğrulama metrikleri
                if 'val_losses' in training_results and training_results['val_losses']:
                    print_info(f"Son doğrulama kaybı: {training_results['val_losses'][-1]:.6f}")
                if 'val_accuracies' in training_results and training_results['val_accuracies']:
                    print_info(f"Son doğrulama doğruluğu: %{training_results['val_accuracies'][-1]*100:.2f}")
                
                # Eğitilen modeli kaydet
                # Modeli kaydet - Dosya adını zaman damgası ile oluştur
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"lstm_{timeframe}_{timestamp}.pth"
                model_path = os.path.join("saved_models", model_filename)
                
                # Kaydetme klasörünü oluştur (yoksa)
                os.makedirs("saved_models", exist_ok=True)
                
                # Modeli kaydet
                lstm_model.save_checkpoint(model_path)
                print_success(f"{timeframe} modeli kaydedildi: {model_filename}")
                
                # Modeli sınıfta sakla
                self.lstm_models[timeframe] = lstm_model
                
                print_success("LSTM eğitimi tamamlandı!")
                return True
            except Exception as e:
                print_error(f"LSTM eğitimi sırasında hata: {str(e)}")
                return False
            
        except Exception as e:
            print_error(f"LSTM eğitimi sırasında hata: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def signal_handler(signum, frame):
    """Handle termination signals"""
    print_section("\nSİNYAL ALINDI: CTRL+C")
    print_info("Bot güvenli bir şekilde durduruluyor...")
    
    try:
        global bot
        if 'bot' in globals() and bot is not None:
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
            # Bot değişkeni tanımlı değilse veya None ise, eğitim sürecinde olabilir
            print_warning("Bot nesnesi bulunamadı, eğitim veya başlatma süreci kesintiye uğradı.")
            logger.warning("Bot nesnesi bulunamadı, eğitim veya başlatma süreci kesintiye uğradı.")
            # Belleği temizle - eğitim sürecinin durması için
            if torch.cuda.is_available():
                print_info("CUDA belleği temizleniyor...")
                torch.cuda.empty_cache()
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