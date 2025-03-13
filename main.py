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
import sys
import MetaTrader5 as mt5  # MetaTrader5 modülünü import et
import concurrent.futures
from utils.colab_manager import ColabManager
import codecs
import psutil

# Windows'da Unicode desteği için
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Renkli çıktı için yardımcı fonksiyonlar
def print_info(message):
    """Bilgi mesajı yazdır"""
    print(f"\u2139 {message}")

def print_success(message):
    """Başarı mesajı yazdır"""
    print(f"\u2705 {message}")  # ✅

def print_error(message):
    """Hata mesajı yazdır"""
    print(f"\u274C {message}")  # ❌

def print_warning(message):
    """Uyarı mesajı yazdır"""
    print(f"\u26A0 {message}")  # ⚠

def print_section(message):
    """Bölüm başlığı yazdır"""
    print("\n" + "="*50)
    print(message)
    print("="*50 + "\n")

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
        
        # Create config directory if it doesn't exist
        config_dir = os.path.join(self.base_dir, 'config')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            print_info("Config dizini oluşturuldu")
        
        # Check for required config files
        required_files = {
            'credentials.json': 'Google Cloud credentials',
            'colab_config.json': 'Colab configuration'
        }
        
        missing_files = []
        for file, description in required_files.items():
            if not os.path.exists(os.path.join(config_dir, file)):
                missing_files.append(f"{file} ({description})")
        
        if missing_files:
            print_warning("Eksik konfigürasyon dosyaları tespit edildi:")
            for file in missing_files:
                print(f"  • {file}")
            print("\nLütfen eksik dosyaları config/ dizinine ekleyin.")
            print("Detaylı bilgi için README.md dosyasına bakın.")
            sys.exit(1)
        
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
        
        # Initialize Colab Manager
        self.colab_manager = None
        
        # Initialize everything
        self.initialize()
        
    def initialize(self):
        """Initializes all components and models"""
        try:
            print_section("BOT BAŞLATILIYOR")
            
            # MT5 bağlantısını kontrol et
            if self.mt5 is None or not self.mt5.connected:
                print_info("MetaTrader 5'e bağlanılıyor...")
                if not self.connect_mt5():
                    raise Exception("MT5 bağlantısı başarısız!")
                print_success("MT5 bağlantısı başarılı!")
            else:
                print_success("MT5 bağlantısı zaten kurulmuş.")
            
            # DataProcessor'ı başlat
            self.data_processor = DataProcessor()
            if not self.data_processor:
                raise Exception("DataProcessor başlatılamadı!")
            
            # Risk Manager'ı başlat
            account_info = self.mt5.get_account_info()
            self.risk_manager = RiskManager(
                initial_balance=account_info.balance,
                risk_per_trade=TRADING_CONFIG.get('risk_per_trade', 0.01),  # Default 1%
                max_daily_loss=TRADING_CONFIG.get('max_daily_loss', 0.05)  # Default 5%
            )
            if not self.risk_manager:
                raise Exception("RiskManager başlatılamadı!")
                
            # System Monitor'ı başlat
            from utils.system_monitor import SystemMonitor
            self.system_monitor = SystemMonitor(self.mt5, emergency_callback=self.handle_emergency)
            if not self.system_monitor:
                raise Exception("SystemMonitor başlatılamadı!")
            
            # MarketHours nesnesini başlat
            self.market_hours = MarketHours()
            if not self.market_hours:
                raise Exception("MarketHours başlatılamadı!")
            
            # Başlangıç testlerini sor
            print_section("BAŞLANGIÇ TESTLERİ")
            if input("Başlangıç testlerini çalıştırmak ister misiniz? (y/n): ").strip().lower() == 'y':
                if not self.run_initial_tests():
                    print_warning("Başlangıç testleri başarısız oldu, devam edilecek...")
            
            # Model eğitimi gerekli mi kontrol et
            print_section("MODEL DURUMU")
            retrain_input = input("Modelleri yeniden eğitmek istiyor musunuz? (y/n): ").strip().lower()
            
            if retrain_input == 'y':
                self.retrain_models = True
                clean_start_input = input("Mevcut modelleri silip sıfırdan başlamak istiyor musunuz? (y/n): ").strip().lower()
                
                if clean_start_input == 'y':
                    self.clear_existing_models = True
                    print_info("Mevcut modeller silinecek ve eğitim sıfırdan başlayacak.")
                    
                    # Mevcut model dosyalarını sil
                    if os.path.exists('saved_models'):
                        for file in os.listdir('saved_models'):
                            if file.endswith('.pth'):
                                os.remove(os.path.join('saved_models', file))
                        print_success("Mevcut modeller silindi!")
                
                # Model eğitimini başlat
                if not self.train_models():
                    print_warning("\nModel eğitimi başarısız oldu! Bot çalışmak için eğitilmiş modellere ihtiyaç duyar.")
                    print_section("MODEL EĞİTİMİ BAŞARISIZ")
                    retry = input("\nYeniden eğitmeyi denemek ister misiniz? (y/n): ").strip().lower()
                    if retry == 'y':
                        print_info("Yeniden eğitim başlatılıyor...")
                        if not self.train_models():
                            print_error("İkinci eğitim denemesi de başarısız oldu!")
                            raise Exception("Model eğitimi başarısız! Bot çalışamaz.")
                        else:
                            print_success("Yeniden eğitim başarılı!")
                    else:
                        print_error("Model eğitimi başarısız! Bot çalışamaz.")
                        raise Exception("Model eğitimi başarısız! Bot çalışamaz.")
                
                # Eğitim tamamlandıktan sonra clear_existing_models'i False yap
                self.clear_existing_models = False
            else:
                # Mevcut modelleri yükle
                if not self.load_or_create_models():
                    print_error("Model yükleme başarısız!")
                    print_info("Bot çalışmak için eğitilmiş modellere ihtiyaç duyar.")
                    raise Exception("Model yükleme başarısız! Bot çalışamaz.")
            
            # MarketHours nesnesini başlat
            self.market_hours = MarketHours()
            if not self.market_hours:
                raise Exception("MarketHours başlatılamadı!")
            
            # RL Trader'ı başlat
            try:
                self.rl_trader = RLTrader(
                    state_size=len(self.data_processor.get_feature_names()),
                    action_size=3,
                    learning_rate=MODEL_CONFIG['rl']['learning_rate']
                )
                print_success("RL Trader başarıyla başlatıldı!")
            except Exception as e:
                print_warning("RL Trader başlatılamadı, yalnızca LSTM ile devam edilecek!")
                self.rl_trader = None
            
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
        print_section("MODEL EĞİTİMİ BAŞLATILIYOR")

        print_info("Eğitim yöntemi seçin:")
        print("\n" + "-"*50)
        print("1) Google Colab (Hızlı, GPU destekli)")
        print("2) Yerel Bilgisayar (Yavaş, CPU)")
        print("-"*50 + "\n")
        
        while True:
            choice = input("Seçiminiz (1/2): ").strip()
            if choice in ['1', '2']:
                break
            print_warning("Lütfen geçerli bir seçim yapın (1 veya 2)")

        if choice == '1':
            print_section("GOOGLE COLAB EĞİTİMİ SEÇİLDİ")
            print_success("Avantajlar:")
            print("  • Daha hızlı eğitim (GPU kullanımı)")
            print("  • Bilgisayarınızı yormaz")
            print("  • Eğitim sırasında bilgisayarınızı kullanabilirsiniz")
            
            print_warning("Gereksinimler:")
            print("  • Google hesabı")
            print("  • İnternet bağlantısı")
            print("  • credentials.json dosyası")
            print("\n" + "-"*50)
            
            success = self._train_models_colab()
            if not success:
                print_section("MODEL EĞİTİMİ BAŞARISIZ")
                print_error("Google Colab üzerinde model eğitimi başarısız oldu!")
                print_info("Lütfen hata mesajlarını kontrol edin ve tekrar deneyin.")
                print_info("Alternatif olarak yerel eğitimi deneyebilirsiniz.")
                return False
            # Eğitim tamamlandıktan sonra clear_existing_models'i False yap
            self.clear_existing_models = False
            return True
        else:
            print_section("YEREL EĞİTİM SEÇİLDİ")
            print_success("Avantajlar:")
            print("  • İnternet bağlantısı gerekmez")
            print("  • Google hesabı gerekmez")
            
            print_warning("Dikkat:")
            print("  • Eğitim süresi uzun olabilir")
            print("  • Bilgisayarınız yoğun çalışacak")
            print("  • Eğitim sırasında bilgisayarınız yavaşlayabilir")
            print("\n" + "-"*50)
            
            success = self._train_models_local()
            if not success:
                print_section("MODEL EĞİTİMİ BAŞARISIZ")
                print_error("Yerel bilgisayarda model eğitimi başarısız oldu!")
                print_info("Lütfen hata mesajlarını kontrol edin ve tekrar deneyin.")
                print_info("Sistem kaynaklarınız yetersiz olabilir, Google Colab eğitimini deneyebilirsiniz.")
                return False
            return True

    def _train_models_colab(self):
        """Google Colab üzerinde modelleri eğit"""
        try:
            print_section("GOOGLE COLAB EĞİTİMİ")
            print_info("Google Colab entegrasyonu başlatılıyor...")
            
            try:
                self.colab_manager = ColabManager()
                # Başlangıç kontrollerini yap
                if not self.colab_manager.config:
                    raise Exception("Colab konfigürasyonu yüklenemedi!")
                if not self.colab_manager.drive_service:
                    raise Exception("Google Drive bağlantısı kurulamadı!")
            except Exception as e:
                print_error("Colab bağlantısı kurulamadı!")
                print(f"\nHata detayı: {str(e)}")
                print("\nÖnerilen çözümler:")
                print("1. credentials.json dosyasının varlığını kontrol edin")
                print("2. İnternet bağlantınızı kontrol edin")
                print("3. Google hesap izinlerinizi kontrol edin")
                return False
            
            print_section("VERİ HAZIRLAMA")
            print_info("Eğitim verisi hazırlanıyor...")
            try:
                data_file = self.prepare_training_data()
                if not data_file:
                    raise Exception("Veri hazırlama başarısız!")
                print_success("Eğitim verisi hazırlandı!")
            except Exception as e:
                print_error("Eğitim verisi hazırlanamadı!")
                print(f"\nHata detayı: {str(e)}")
                return False
            
            print_section("GOOGLE DRIVE YÜKLEME")
            print_info("Veriler Google Drive'a yükleniyor...")
            try:
                if not self.colab_manager.upload_data_to_drive(data_file):
                    raise Exception("Drive'a yükleme başarısız!")
                print_success("Veriler başarıyla yüklendi ve Drive'da doğrulandı!")
            except Exception as e:
                print_error("Veriler Google Drive'a yüklenemedi!")
                print(f"\nHata detayı: {str(e)}")
                print("\nÖnerilen çözümler:")
                print("1. Google Drive'da yeterli alan olduğunu kontrol edin")
                print("2. İnternet bağlantınızı kontrol edin")
                print("3. Google Drive API izinlerini kontrol edin")
                print("4. Drive'daki trading_bot/data klasörünün varlığını kontrol edin")
                print("5. colab_config.json dosyasındaki klasör ID'lerini kontrol edin")
                return False
            
            print_section("EĞİTİM BAŞLATMA")
            print_info("Colab'da eğitim başlatılıyor...")
            try:
                # Colab notebook ID'sini kontrol et
                if not self.colab_manager.colab_notebook_id or len(self.colab_manager.colab_notebook_id) < 20:
                    print_error("Geçersiz Colab Notebook ID!")
                    print("\nHata detayı: config/colab_config.json dosyasındaki ID'ler geçersiz.")
                    print("\nÇözüm için:")
                    print("1. Google Drive'a bir Colab notebook yükleyin")
                    print("2. Notebook'un ID'sini alın (URL'deki /notebooks/... kısmından sonraki ID)")
                    print("3. config/colab_config.json dosyasını düzenleyin ve doğru ID'yi ekleyin")
                    print("4. Aynı şekilde model ve veri klasörlerinin ID'lerini de güncelleyin")
                    return False
                
                if not self.colab_manager.start_colab_training():
                    raise Exception("Eğitim başlatılamadı!")
                print_success("Eğitim başarıyla başlatıldı!")
            except Exception as e:
                print_error("Colab'da eğitim başlatılamadı!")
                print(f"\nHata detayı: {str(e)}")
                print("\nÖnerilen çözümler:")
                print("1. Colab notebook'un Drive'da olduğunu kontrol edin")
                print("2. Notebook'un doğru klasörde olduğunu kontrol edin")
                print("3. colab_config.json dosyasını kontrol edin")
                return False
            
            print_section("EĞİTİM TAMAMLAMA")
            print_info("Eğitimin tamamlanması bekleniyor...")
            try:
                if not self.colab_manager.wait_for_training_completion():
                    raise Exception("Eğitim tamamlanamadı!")
                print_success("Eğitim başarıyla tamamlandı!")
            except Exception as e:
                print_error("Eğitim tamamlanamadı!")
                print(f"\nHata detayı: {str(e)}")
                print("\nÖnerilen çözümler:")
                print("1. Colab oturumunun aktif olduğunu kontrol edin")
                print("2. İnternet bağlantınızı kontrol edin")
                print("3. Colab'ın bağlantısının kesilmediğinden emin olun")
                return False
            
            print_section("MODEL İNDİRME")
            print_info("Eğitilen model indiriliyor...")
            try:
                if not self.colab_manager.download_model():
                    raise Exception("Model indirme başarısız!")
                print_success("Model başarıyla indirildi!")
            except Exception as e:
                print_error("Model indirilemedi!")
                print(f"\nHata detayı: {str(e)}")
                print("\nÖnerilen çözümler:")
                print("1. Drive'da model dosyasının oluştuğunu kontrol edin")
                print("2. İnternet bağlantınızı kontrol edin")
                print("3. Drive API izinlerini kontrol edin")
                return False
            
            print_section("EĞİTİM TAMAMLANDI")
            print_success("Google Colab eğitimi başarıyla tamamlandı!")
            return True
            
        except Exception as e:
            logger.error(f"Colab eğitim hatası: {str(e)}")
            print_error("Beklenmeyen bir hata oluştu!")
            print(f"\nHata detayı: {str(e)}")
            print("\nLütfen tüm adımları kontrol edip tekrar deneyin.")
            return False

    def _train_models_local(self):
        """Yerel bilgisayarda modelleri eğit"""
        try:
            print_section("YEREL EĞİTİM BAŞLATILIYOR")
            print_info("Yerel bilgisayarda model eğitimi başlatılıyor...")
            
            # Başarılı eğitilen model sayısı
            successful_models = 0
            
            # Her zaman dilimi için ayrı eğitim
            for timeframe in self.timeframes:
                print_section(f"{timeframe.upper()} MODELİ EĞİTİMİ")
                print_info(f"{timeframe} zaman dilimi için LSTM modeli eğitiliyor...")
                print_warning("Bu işlem birkaç saat sürebilir, lütfen bekleyin...")
                
                try:
                    success = self.train_lstm_model(timeframe)
                    if success:
                        print_success(f"{timeframe} modeli başarıyla eğitildi!")
                        successful_models += 1
                    else:
                        print_error(f"{timeframe} modeli eğitimi başarısız oldu!")
                        print_info("Diğer zaman dilimlerine geçiliyor...")
                except Exception as e:
                    logger.error(f"{timeframe} modeli eğitilirken hata: {str(e)}")
                    print_error(f"{timeframe} modeli eğitilirken hata: {str(e)}")
                    print_info("Diğer zaman dilimlerine geçiliyor...")
            
            # En az bir model başarıyla eğitildi mi kontrol et
            if not any(self.lstm_models.values()):
                print_section("EĞİTİM BAŞARISIZ")
                print_error("Hiçbir model başarıyla eğitilemedi!")
                return False
                
            print_section("EĞİTİM TAMAMLANDI")
            print_success(f"Toplam {successful_models} model başarıyla eğitildi!")
            # Eğitim tamamlandıktan sonra clear_existing_models'i False yap
            self.clear_existing_models = False
            return successful_models > 0
            
        except Exception as e:
            logger.error(f"Yerel eğitim hatası: {str(e)}")
            print_error("Yerel eğitim hatası!")
            print(f"\nHata detayı: {str(e)}")
            print("\nÖnerilen çözümler:")
            print("1. Sistem kaynaklarınızın yeterli olduğundan emin olun")
            print("2. MT5 bağlantınızı kontrol edin")
            print("3. Veri kaynaklarınızı kontrol edin")
            print("4. Google Colab eğitimini deneyin (daha hızlı ve güvenilir)")
            return False

    def get_trading_signals(self, timeframe):
        """
        Belirli bir zaman dilimi için ticaret sinyalleri üretir
        
        Parametreler:
        - timeframe: Zaman dilimi (örn. '5m', '15m', '1h')
        
        Dönüş:
        - lstm_prediction: LSTM modeli tahmini
        - rl_action: RL modeli aksiyonu
        """
        try:
            # Bellek kullanımını kontrol et
            memory_usage = self.get_memory_usage()
            if memory_usage > 1024:  # 1GB'dan fazla kullanım varsa
                logger.warning(f"Yüksek bellek kullanımı: {memory_usage:.1f} MB")
                gc.collect()  # Garbage collection'ı zorla
            
            logger.debug(f"{timeframe} zaman dilimi için ticaret sinyalleri üretiliyor...")
            
            # Veri doğrulama - 1: Timeframe kontrolü
            valid_timeframes = ['5m', '15m', '30m', '1h', '4h', 'D1']
            if timeframe not in valid_timeframes:
                logger.error(f"Geçersiz zaman dilimi: {timeframe}")
                return None, None
            
            # Son 100 mum verisini al - chunk'lar halinde
            chunk_size = 50
            chunks = []
            
            for i in range(0, 100, chunk_size):
                chunk = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=chunk_size, start_pos=i)
                if chunk is None or len(chunk) == 0:
                    logger.error(f"Veri chunk'ı alınamadı: {i}-{i+chunk_size}")
                    return None, None
                chunks.append(chunk)
            
            # Chunk'ları birleştir
            candles = pd.concat(chunks, ignore_index=True)
            
            # Veri doğrulama - 2: Minimum veri kontrolü
            if len(candles) < 60:
                logger.warning(f"{timeframe} için yeterli veri yok. En az 60 mum gerekli.")
                return None, None
            
            # Veri doğrulama - 3: Sütun kontrolü
            required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
            missing_columns = [col for col in required_columns if col not in candles.columns]
            
            if missing_columns:
                logger.warning(f"{timeframe} verisinde eksik sütunlar: {missing_columns}")
                logger.debug(f"Mevcut sütunlar: {list(candles.columns)}")
                
                if 'tick_volume' in missing_columns and 'volume' in candles.columns:
                    logger.debug("'volume' sütunu 'tick_volume' olarak kullanılıyor")
                    candles['tick_volume'] = candles['volume']
                    missing_columns.remove('tick_volume')
                
                if missing_columns:
                    logger.error(f"Kritik sütunlar eksik: {missing_columns}")
                    return None, None
            
            # Veri doğrulama - 4: NaN kontrolü
            nan_columns = candles.columns[candles.isna().any()].tolist()
            if nan_columns:
                logger.warning(f"NaN değerler içeren sütunlar: {nan_columns}")
                # Basit forward fill ile doldur
                candles = candles.ffill()
            
            # Veri doğrulama - 5: Aykırı değer kontrolü
            for col in ['open', 'high', 'low', 'close']:
                mean = candles[col].mean()
                std = candles[col].std()
                outliers = candles[col][(candles[col] < mean - 3*std) | (candles[col] > mean + 3*std)]
                if not outliers.empty:
                    logger.warning(f"Aykırı değerler bulundu - {col}: {len(outliers)} adet")
            
            # Teknik göstergeleri paralel olarak hesapla
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_technicals = executor.submit(self.data_processor.add_technical_indicators, candles.copy())
                future_gaps = executor.submit(self.data_processor.detect_price_gaps, candles.copy())
                future_sessions = executor.submit(self.data_processor.add_session_info, candles.copy())
                
                try:
                    candles = future_technicals.result()
                    if candles is None:
                        logger.error("Teknik göstergeler hesaplanamadı")
                        return None, None
                        
                    # Gap ve seans bilgilerini birleştir
                    gap_data = future_gaps.result()
                    session_data = future_sessions.result()
                    
                    if gap_data is not None:
                        candles['gap'] = gap_data['gap']
                        candles['gap_size'] = gap_data['gap_size']
                    else:
                        candles['gap'] = 0
                        candles['gap_size'] = 0
                        
                    if session_data is not None:
                        candles['session_asia'] = session_data['session_asia']
                        candles['session_europe'] = session_data['session_europe']
                        candles['session_us'] = session_data['session_us']
                    else:
                        candles['session_asia'] = 0
                        candles['session_europe'] = 0
                        candles['session_us'] = 0
                        
                except Exception as e:
                    logger.error(f"Paralel işlem hatası: {str(e)}")
                    return None, None
            
            # LSTM tahmini için veriyi hazırla
            try:
                # Boş DataFrame kontrolü
                if candles is None or candles.empty:
                    logger.error("LSTM tahmini için hazırlanan veri boş veya None")
                    return None, None
                
                # Veri hazırlama işlemini chunk'lara böl
                chunk_size = 1000
                num_chunks = (len(candles) + chunk_size - 1) // chunk_size
                prepared_chunks = []
                
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(candles))
                    chunk = candles.iloc[start_idx:end_idx]
                    prepared_chunk = self.data_processor.prepare_prediction_data(chunk)
                    if prepared_chunk is not None:
                        prepared_chunks.append(prepared_chunk)
                
                if not prepared_chunks:
                    logger.error("Hiçbir veri chunk'ı hazırlanamadı")
                    return None, None
                
                lstm_data = torch.cat(prepared_chunks, dim=0)
                
                # LSTM modelini kullanarak tahmin yap
                lstm_model = self.lstm_models.get(timeframe)
                if lstm_model is None:
                    logger.warning(f"{timeframe} için LSTM modeli bulunamadı")
                    return None, None
                
                # Batch'ler halinde tahmin yap
                batch_size = 32
                predictions = []
                
                for i in range(0, len(lstm_data), batch_size):
                    batch = lstm_data[i:i+batch_size]
                    with torch.no_grad():
                        pred = lstm_model.forward(batch)
                        predictions.append(pred)
                
                lstm_prediction = torch.cat(predictions, dim=0).mean().item()
                
                # Tahmin edilen değeri orijinal ölçeğe dönüştür
                lstm_prediction = self.data_processor.inverse_transform_price(lstm_prediction)
                
                # Son kapanış fiyatı
                last_close = candles['close'].iloc[-1]
                
                # Yön ve yüzde değişim
                direction = "YUKARI" if lstm_prediction > last_close else "AŞAĞI"
                change_pct = abs(lstm_prediction - last_close) / last_close * 100
                
                logger.info(f"{timeframe} LSTM Tahmini: {lstm_prediction:.2f} ({direction}, %{change_pct:.2f})")
                
                # RL durumunu hazırla - paralel işlem
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_rl = executor.submit(self.data_processor.prepare_rl_state, candles)
                    rl_state = future_rl.result()
                
                # RL modelini kullanarak aksiyon al
                rl_model = self.rl_trader
                if rl_model is None:
                    logger.debug(f"{timeframe} için RL modeli bulunamadı")
                    rl_action = None
                else:
                    rl_action = rl_model.predict(rl_state)
                    logger.info(f"{timeframe} RL Aksiyonu: {rl_action}")
                
                # Belleği temizle
                del candles, lstm_data, predictions
                gc.collect()
                
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
            print_section("TİCARET YÜRÜTME BİTİŞ")
        
    def run(self):
        """Runs the Trading Bot"""
        try:
            print_section("XAUUSD TRADING BOT BAŞLATILIYOR")
            print("Bot'u durdurmak için Ctrl+C tuşlarına basın")
            print_section("SİSTEM KONTROLÜ")
            
            # Sistem kontrollerini yap
            if not self.system_check():
                raise Exception("Sistem kontrolleri başarısız!")
            
            # Model kontrolü
            if not any(self.lstm_models.values()):
                print_warning("LSTM modelleri bulunamadı veya yüklenemedi!")
                if input("Modelleri yeniden eğitmek ister misiniz? (y/n): ").strip().lower() == 'y':
                    if not self.train_models():
                        raise Exception("Model eğitimi başarısız!")
                else:
                    raise Exception("Çalışır durumda model bulunamadı!")
            
            if not self.rl_trader:
                print_warning("RL modeli bulunamadı veya düzgün yüklenemedi!")
                print_info("Bot yalnızca LSTM modeli kullanarak çalışacak.")
                print_info("Daha iyi sonuçlar için, bot'u durdurup modelleri yeniden eğitebilirsiniz.")
            
            # Trading döngüsünü başlat
            print_section("TRADING DÖNGÜSÜ BAŞLIYOR")
            self.is_running = True
            
            while self.is_running:
                try:
                    # Piyasa açık mı kontrol et
                    if not self.market_hours.is_market_open():
                        print_info("Piyasa kapalı. Bir sonraki açılışı bekleniyor...")
                        time.sleep(60)  # 1 dakika bekle
                        continue
                    
                    # Her timeframe için işlem yap
                    for timeframe in self.timeframes:
                        self.execute_trades(timeframe)
                    
                    # Sistem durumunu kontrol et
                    if not self.system_monitor.check_status():
                        print_warning("Sistem durumu kritik! Yeniden başlatılıyor...")
                        self.restart()
                    
                    # Belleği temizle
                    gc.collect()
                    
                    # Kısa bir süre bekle
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Trading döngüsü hatası: {str(e)}")
                    print_error(f"Trading döngüsü hatası: {str(e)}")
                    print_info("3 saniye sonra yeniden denenecek...")
                    time.sleep(3)
            
        except KeyboardInterrupt:
            print_info("\nBot kullanıcı tarafından durduruldu.")
        except Exception as e:
            logger.error(f"Kritik hata: {str(e)}")
            print_error(f"Kritik hata: {str(e)}")
        finally:
            self.cleanup()
            print_section("BOT DURDURULDU")

    def system_check(self):
        """Sistem durumunu kontrol et"""
        try:
            # MT5 bağlantısı
            if not self.mt5 or not self.mt5.connected:
                print_error("MT5 bağlantısı yok!")
                return False
            
            # Sistem durumunu kontrol et
            if self.system_monitor:
                stats = self.system_monitor.get_system_stats()
                
                # Bellek durumu
                if stats['memory_usage'] > 90:
                    print_error(f"Yüksek bellek kullanımı! (%{stats['memory_usage']:.1f})")
                    return False
                
                # CPU durumu
                if stats['cpu_usage'] > 95:
                    print_error(f"Yüksek CPU kullanımı! (%{stats['cpu_usage']:.1f})")
                    return False
                
                # Bağlantı durumu
                if not stats['connection_status']:
                    print_error("MT5 bağlantısı koptu!")
                    return False
            else:
                print_warning("Sistem monitörü başlatılmamış!")
            
            print_success("Sistem kontrolleri başarılı!")
            return True
            
        except Exception as e:
            logger.error(f"Sistem kontrol hatası: {str(e)}")
            print_error(f"Sistem kontrol hatası: {str(e)}")
            return False

    def cleanup(self):
        """Kaynakları temizle"""
        try:
            # MT5 bağlantısını kapat
            if self.mt5:
                self.mt5.shutdown()
                print_info("MT5 bağlantısı kapatıldı.")
            
            # Belleği temizle
            gc.collect()
            print_info("Bellek temizlendi.")
            
            # Log dosyalarını kapat
            logging.shutdown()
            print_info("Log sistemi kapatıldı.")
            
        except Exception as e:
            logger.error(f"Temizleme hatası: {str(e)}")
            print_error(f"Temizleme hatası: {str(e)}")
            
    def restart(self):
        """Bot'u yeniden başlat"""
        try:
            print_section("BOT YENİDEN BAŞLATILIYOR")
            logger.warning("Bot yeniden başlatılıyor...")
            
            # Mevcut durumu kaydet
            self.is_running = False
            
            # Açık pozisyonları kontrol et
            if self.mt5 and self.mt5.connected:
                positions = self.mt5.get_open_positions()
                if positions:
                    print_warning(f"{len(positions)} açık pozisyon bulundu.")
                    print_info("Pozisyonlar korunacak ve bot yeniden başlatıldıktan sonra izlenecek.")
            
            # MT5 bağlantısını yenile
            if self.mt5:
                self.mt5.disconnect()
                time.sleep(2)  # Bağlantının tamamen kapanması için bekle
                
                # Yeniden bağlan
                print_info("MT5 bağlantısı yenileniyor...")
                if not self.connect_mt5():
                    raise Exception("MT5 bağlantısı yenilenemedi!")
                print_success("MT5 bağlantısı yenilendi!")
            
            # Belleği temizle
            gc.collect()
            
            # Sistem monitörünü yeniden başlat
            if self.system_monitor:
                self.system_monitor.stop_monitoring()
                from utils.system_monitor import SystemMonitor
                self.system_monitor = SystemMonitor(self.mt5, emergency_callback=self.handle_emergency)
                self.system_monitor.start_monitoring()
                print_success("Sistem monitörü yeniden başlatıldı!")
            
            # Bot'u yeniden çalıştır
            print_success("Bot yeniden başlatıldı!")
            self.is_running = True
            
            return True
            
        except Exception as e:
            logger.error(f"Bot yeniden başlatma hatası: {str(e)}")
            print_error(f"Bot yeniden başlatma hatası: {str(e)}")
            return False

    def get_memory_usage(self):
        """Return current memory usage in MB"""
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
        """Logger'ı yapılandır"""
        try:
            # Log dizinini oluştur
            os.makedirs('logs', exist_ok=True)
            
            # Dosya adını oluştur
            log_filename = f'logs/trading_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            
            # Root logger'ı yapılandır
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            
            # Konsol handler'ı oluştur ve yapılandır
            if sys.platform == 'win32':
                # Windows için özel handler - reconfigure kullanmadan
                # sys.stdout.reconfigure(encoding='utf-8') - Bu satır hataya neden oluyor
                console_handler = logging.StreamHandler(sys.stdout)
            else:
                console_handler = logging.StreamHandler(sys.stdout)
            
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            
            # Dosya handler'ı oluştur ve yapılandır
            file_handler = logging.FileHandler(log_filename, 'a', 'utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            file_handler.setFormatter(file_formatter)
            
            # Mevcut handler'ları temizle
            root_logger.handlers.clear()
            
            # Yeni handler'ları ekle
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)
            
            # Trading bot logger'ını yapılandır
            self.logger = logging.getLogger("TradingBot")
            self.logger.setLevel(logging.INFO)
            
            self.logger.info("Logger başarıyla yapılandırıldı")
            return True
            
        except Exception as e:
            print(f"Logger yapılandırma hatası: {str(e)}")
            return False

    def train_lstm_model(self, timeframe):
        """
        Belirli bir zaman dilimi için LSTM modelini eğitir
        
        Parametreler:
        - timeframe: Zaman dilimi (örn. '5m', '15m', '1h')
        
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
            sequence_length = self.config['MODEL']['training']['sequence_length']
            train_split = self.config['MODEL']['training']['train_split']
            weight_recent_factor = self.config['MODEL']['weight_recent_factor'] if 'weight_recent_factor' in self.config['MODEL'] else 2.0
            
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
            hidden_size = self.config['MODEL']['lstm']['hidden_size']
            num_layers = self.config['MODEL']['lstm']['num_layers']
            dropout = self.config['MODEL']['lstm']['dropout']
            learning_rate = self.config['MODEL']['training']['learning_rate'] if 'learning_rate' in self.config['MODEL']['training'] else 0.001
            batch_size = self.config['MODEL']['training']['batch_size'] if 'batch_size' in self.config['MODEL']['training'] else 64
            epochs = self.config['MODEL']['training']['epochs'] if 'epochs' in self.config['MODEL']['training'] else 100
            patience = self.config['MODEL']['training']['early_stopping_patience'] if 'early_stopping_patience' in self.config['MODEL']['training'] else 15
            
            # Modeli oluştur
            # LSTMPredictor sınıfı config parametresi bekliyor, o yüzden config oluşturalım
            # MODEL_CONFIG içindeki mevcut parametreleri kullanıyoruz
            lstm_config = {
                'lstm': {
                    'input_size': input_size,  # Bu dinamik olarak hesaplanıyor
                    'hidden_size': MODEL_CONFIG['lstm']['hidden_size'],
                    'num_layers': MODEL_CONFIG['lstm']['num_layers'],
                    'dropout': MODEL_CONFIG['lstm']['dropout'],
                    'bidirectional': MODEL_CONFIG['lstm']['bidirectional']
                },
                'batch_norm': MODEL_CONFIG['batch_norm'],
                'attention': MODEL_CONFIG['attention']
            }
            model = LSTMPredictor(config=lstm_config)
            
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
            model_filename = f"lstm_{timeframe}"
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

    def stop(self):
        """Bot'u güvenli bir şekilde durdur"""
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

    def prepare_training_data(self):
        """Prepare and save training data"""
        try:
            # Veriyi hazırla
            data = {}
            for timeframe in self.timeframes:
                candles = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=DATA_CONFIG['training_candles'][timeframe])
                if candles is None:
                    raise Exception(f"Veri çekme hatası: {timeframe}")
                data[timeframe] = candles
                
            # Veriyi işle
            processed_data = self.data_processor.process_training_data(data)
            
            # CSV olarak kaydet
            save_path = "data/training_data.csv"
            processed_data.to_csv(save_path, index=False)
            
            return save_path
            
        except Exception as e:
            logger.error(f"Veri hazırlama hatası: {str(e)}")
            return None

    def connect_mt5(self):
        """MT5'e bağlan ve hesap bilgilerini kontrol et"""
        try:
            # Eğer MT5 bağlantısı zaten kurulmuşsa, tekrar bağlanmaya çalışma
            if self.mt5 is not None and self.mt5.connected:
                print_info("MT5 bağlantısı zaten kurulmuş.")
                return True
                
            # MT5 bağlantısını oluştur
            self.mt5 = MT5Connector(
                login=self.config['MT5']['login'],
                password=self.config['MT5']['password'],
                server=self.config['MT5']['server']
            )
            
            # Bağlantı kontrolü
            if not self.mt5.connected:
                print_error("MT5 bağlantısı kurulamadı!")
                print_info("Lütfen MT5 terminalinin açık olduğundan emin olun.")
                return False
            
            # Hesap bilgilerini kontrol et
            account_info = self.mt5.get_account_info()
            if not account_info:
                print_error("Hesap bilgileri alınamadı!")
                return False
            
            # Sembol bilgilerini kontrol et
            symbol_info = self.mt5.symbol_info("XAUUSD")
            if symbol_info is None:
                print_warning("XAUUSD sembolü bulunamadı!")
                print_info("Lütfen sembolün doğru olduğundan emin olun.")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 bağlantı hatası: {str(e)}")
            print_error(f"MT5 bağlantı hatası: {str(e)}")
            return False

    def run_initial_tests(self):
        """Başlangıç testlerini çalıştır"""
        try:
            print_info("Başlangıç testleri çalıştırılıyor...")
            
            # MT5 bağlantı testi
            if not self.mt5 or not self.mt5.connected:
                print_error("MT5 bağlantı testi başarısız!")
                return False
            print_success("MT5 bağlantı testi başarılı")
            
            # Veri çekme testi
            for timeframe in self.timeframes:
                print_info(f"{timeframe} için veri çekme testi...")
                data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=100)
                if data is None or len(data) < 100:
                    print_error(f"{timeframe} veri çekme testi başarısız!")
                    return False
                print_success(f"{timeframe} veri çekme testi başarılı")
            
            # Teknik gösterge hesaplama testi
            print_info("Teknik gösterge hesaplama testi...")
            if not hasattr(self, 'data_processor'):
                self.data_processor = DataProcessor()
            
            test_data = self.mt5.get_historical_data("XAUUSD", "5m", num_candles=100)
            if test_data is None:
                print_error("Test verisi alınamadı!")
                return False
            
            processed_data = self.data_processor.add_technical_indicators(test_data)
            if processed_data is None:
                print_error("Teknik gösterge hesaplama testi başarısız!")
                return False
            print_success("Teknik gösterge hesaplama testi başarılı")
            
            # Risk yönetimi testi
            print_info("Risk yönetimi testi...")
            if not hasattr(self, 'risk_manager'):
                self.risk_manager = RiskManager(initial_balance=100000)
            
            # Test trade parameters
            entry_price = 2000
            stop_loss = 1990
            
            # Test position size calculation
            lot_size = self.risk_manager.calculate_position_size(entry_price, stop_loss)
            if lot_size <= 0:
                print_error("Risk yönetimi testi başarısız - Lot hesaplama hatası!")
                return False
                
            # Test trading permission
            if not self.risk_manager.can_trade():
                print_error("Risk yönetimi testi başarısız - İşlem izni hatası!")
                return False
                
            print_success("Risk yönetimi testi başarılı")
            
            print_success("Tüm başlangıç testleri başarılı!")
            return True
            
        except Exception as e:
            logger.error(f"Test hatası: {str(e)}")
            print_error(f"Test hatası: {str(e)}")
            return False

    def check_system_resources(self):
        """Sistem kaynaklarını kontrol et"""
        try:
            # Bellek kullanımını kontrol et
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB cinsinden
            
            # CPU kullanımını kontrol et
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk alanını kontrol et
            disk_usage = psutil.disk_usage('/').percent
            
            self.logger.info(f"""
==================================================
SİSTEM KAYNAKLARI
==================================================
Bellek Kullanımı: {memory_usage:.2f} MB
CPU Kullanımı: {cpu_percent:.1f}%
Disk Kullanımı: {disk_usage:.1f}%
==================================================
""")
            
            # Eşik değerlerini kontrol et
            if memory_usage > 1024:  # 1 GB
                self.logger.warning("⚠ Yüksek bellek kullanımı!")
            if cpu_percent > 80:
                self.logger.warning("⚠ Yüksek CPU kullanımı!")
            if disk_usage > 90:
                self.logger.warning("⚠ Disk alanı kritik seviyede!")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Sistem kontrol hatası: {str(e)}")
            return False

    def shutdown(self):
        """Bot'u güvenli bir şekilde durdur"""
        try:
            self.logger.info("\n==================================================")
            self.logger.info("BOT DURDURULDU")
            self.logger.info("==================================================\n")
            
            # MT5 bağlantısını kapat
            if hasattr(self, 'mt5_connector'):
                self.mt5_connector.disconnect()
            
            # Açık dosyaları kapat
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
            
            return True
            
        except Exception as e:
            print(f"\nTemizleme hatası: {str(e)}")
            return False

def signal_handler(signum, frame):
    """Sinyal yöneticisi"""
    print("\nKapatma sinyali alındı. Bot güvenli bir şekilde kapatılacak...")
    try:
        if 'bot' in globals() and bot is not None:
            bot.stop()
    except Exception as e:
        print(f"Bot kapatılırken hata oluştu: {str(e)}")
    finally:
        sys.exit(0)

def main():
    """Ana program"""
    global bot  # bot değişkenini global olarak tanımla
    bot = None  # Başlangıçta None olarak ayarla
    
    try:
        # Sinyal yöneticilerini ayarla
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        print_section("ALTIN TİCARET BOTUNA HOŞGELDİNİZ")
        print_info("Bot başlatılıyor...")

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
        print_section("MT5 BAĞLANTISI")
        print_info("MetaTrader 5'e bağlanılıyor...")
        
        mt5_connector = MT5Connector(
            login=MT5_CONFIG['login'],
            password=MT5_CONFIG['password'],
            server=MT5_CONFIG['server']
        )
        
        # Bağlantı kontrolü
        if not mt5_connector.connected:
            print_error("MT5 bağlantısı kurulamadı!")
            print_info("Lütfen MT5 terminalinin açık olduğundan emin olun.")
            print_section("PROGRAM SONLANDIRILIYOR")
            return
        
        # Hesap bilgilerini göster
        account_info = mt5_connector.get_account_info()
        if account_info:
            print_success("Bağlantı başarılı!")
            print_info(f"Hesap: {account_info.login}")
            print_info(f"Sunucu: {account_info.server}")
            print_info(f"Bakiye: ${account_info.balance:.2f}")
            print_info(f"Özsermaye: ${account_info.equity:.2f}")
            print_info(f"Marjin: ${account_info.margin:.2f}")
            print_info(f"Serbest Marjin: ${account_info.margin_free:.2f}")
        
        # Sembol bilgilerini kontrol et
        symbol_info = mt5_connector.symbol_info("XAUUSD")
        if symbol_info is None:
            print_warning("XAUUSD sembolü bulunamadı!")
            print_info("Lütfen sembolün doğru olduğundan emin olun.")
            print_section("PROGRAM SONLANDIRILIYOR")
            return
        
        print_section("SEMBOL BİLGİLERİ")
        print_info("XAUUSD sembol bilgileri:")
        print_info(f"Pip değeri: {symbol_info.point}")
        print_info(f"Spread: {symbol_info.spread} puan")
        print_info(f"Minimum lot: {symbol_info.volume_min}")
        print_info(f"Maksimum lot: {symbol_info.volume_max}")
        print_info(f"Lot adımı: {symbol_info.volume_step}")
        
        # Veri işleyici oluştur
        data_processor = DataProcessor()
        
        # Risk yöneticisi oluştur
        risk_manager = RiskManager(mt5_connector)
        
        # Bot oluştur ve MT5 bağlantısını aktar
        bot = XAUUSDTradingBot()
        bot.mt5 = mt5_connector  # Mevcut MT5 bağlantısını bot nesnesine aktar
        
        # Modelleri başlat
        print_section("MODEL KONTROLÜ")
        print_info("Modeller yükleniyor...")
        
        try:
            models_loaded = bot.load_or_create_models()
            
            # Modeller bulunamadıysa veya yüklenemediyse
            if not models_loaded:
                print_error("Model bulunamadı veya yüklenemedi!")
                print_info("Bot çalışmak için eğitilmiş modellere ihtiyaç duyar.")
                
                # Kullanıcıya eğitim isteyip istemediğini sor
                user_input = input("\nModelleri eğitmek ister misiniz? Bu işlem zaman alabilir. (y/n): ").strip().lower()
                
                if user_input == 'y':
                    print_section("MODEL EĞİTİMİ")
                    print_info("Modeller eğitiliyor... Bu işlem zaman alabilir.")
                    
                    training_success = bot.train_models()
                    
                    if not training_success:
                        print_error("Model eğitimi başarısız oldu!")
                        print_info("Bot çalışmak için eğitilmiş modellere ihtiyaç duyar.")
                        print_section("PROGRAM SONLANDIRILIYOR")
                        return
                        
                    print_success("Modeller eğitildi, bot başlatılıyor...")
                else:
                    print_warning("Modeller eğitilmeden bot düzgün çalışamayacak.")
                    print_section("PROGRAM SONLANDIRILIYOR")
                    return
                
        except Exception as e:
            print_error(f"Model yükleme hatası: {str(e)}")
            print_section("PROGRAM SONLANDIRILIYOR")
            print_info("Modeller yüklenemedi, program sonlandırılıyor...")
            return
        
        # Bot'u başlat
        print_section("BOT BAŞLATILIYOR")
        print_success("Tüm sistemler hazır!")
        
        # Sistem monitörünü başlat
        if hasattr(bot, 'system_monitor') and bot.system_monitor:
            bot.system_monitor.start_monitoring()
            print_success("Sistem monitörü başlatıldı!")
        
        # Bot'u çalıştır
        bot.run()
        
    except KeyboardInterrupt:
        print_info("\nBot kullanıcı tarafından durduruldu.")
        if 'bot' in locals() and bot is not None:
            bot.stop()
    except Exception as e:
        logger.error(f"Kritik hata: {str(e)}")
        print_error(f"Kritik hata oluştu: {str(e)}")
        if 'bot' in locals() and bot is not None:
            bot.stop()
    finally:
        print_section("PROGRAM SONLANDIRILDI")

if __name__ == "__main__":
    main() 