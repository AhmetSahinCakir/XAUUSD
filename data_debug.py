import os
import sys
import pandas as pd
import time
import logging
import torch
import numpy as np
from datetime import datetime
import psutil
import json

# Sistem yolunu ayarla
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MT5 bağlantısı ve veri işleme
from utils.mt5_connector import MT5Connector
from utils.data_processor import DataProcessor

# Logger ayarları
logger = logging.getLogger("TradingBot.DataDebug")

class DataDebugger:
    def __init__(self, mt5_connector=None, data_processor=None):
        """
        Veri ve bağlantı debug işlemleri için sınıf
        
        Parametreler:
        - mt5_connector: Varolan MT5Connector nesnesi (opsiyonel)
        - data_processor: Varolan DataProcessor nesnesi (opsiyonel)
        """
        if mt5_connector and mt5_connector.connected:
            self.mt5 = mt5_connector
        else:
            self.mt5 = MT5Connector()
            
        self.data_processor = data_processor if data_processor else DataProcessor()
        self.results_file = "debug_results.json"
        self.results = {
            'tests': {
                'mt5_connection': {},
                'timeframes': {}
            }
        }
        
    def get_memory_usage(self):
        """Mevcut bellek kullanımını MB cinsinden döndürür"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_mt5_connection(self):
        """MT5 bağlantısını test eder"""
        try:
            if not self.mt5.connected and not self.mt5.connect():
                logger.error("MT5 bağlantısı başarısız!")
                self.results['tests']['mt5_connection'] = {'success': False}
                return
                
            account_info = self.mt5.get_account_info()
            if not account_info:
                logger.error("Hesap bilgileri alınamadı")
                self.results['tests']['mt5_connection'] = {'success': False, 'error': "Hesap bilgileri alınamadı"}
                return
            
            logger.info(f"MT5 bağlantısı başarılı. Hesap: {account_info.login}")
            self.results['tests']['mt5_connection'] = {
                "login": account_info.login,
                "server": account_info.server,
                "balance": account_info.balance,
                "equity": account_info.equity
            }
        except Exception as e:
            logger.error(f"MT5 bağlantı testi sırasında hata: {str(e)}")
            self.results['tests']['mt5_connection'] = {'success': False, 'error': str(e)}
    
    def test_data_retrieval(self, timeframe, num_candles):
        """Belirli bir zaman dilimi için veri alımını test eder"""
        try:
            initial_memory = self.get_memory_usage()
            start_time = time.time()
            
            data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
            
            if data is None:
                self.results['tests']['timeframes'][timeframe]['data_retrieval'] = {
                    'success': False,
                    'error': f"Veri alınamadı - {timeframe}"
                }
                return False, None
            
            # Veri doğrulama
            required_columns = ['open', 'high', 'low', 'close', 'time']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.results['tests']['timeframes'][timeframe]['data_retrieval'] = {
                    'success': False,
                    'error': f"Eksik sütunlar: {missing_columns}"
                }
                return False, None
            
            # Performans metrikleri
            end_time = time.time()
            memory_used = self.get_memory_usage() - initial_memory
            time_taken = end_time - start_time
            
            self.results['tests']['timeframes'][timeframe]['data_retrieval'] = {
                'success': True,
                'rows': len(data),
                'columns': list(data.columns),
                'memory_mb': round(memory_used, 2),
                'time_seconds': round(time_taken, 2),
                'first_date': data['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                'last_date': data['time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return True, data
            
        except Exception as e:
            logger.error(f"Veri alımı testi sırasında hata: {str(e)}")
            self.results['tests']['timeframes'][timeframe]['data_retrieval'] = {
                'success': False,
                'error': str(e)
            }
            return False, None
    
    def test_technical_indicators(self, data, timeframe):
        """Teknik göstergelerin hesaplanmasını test eder"""
        try:
            initial_memory = self.get_memory_usage()
            start_time = time.time()
            
            if 'tick_volume' not in data.columns:
                logger.warning("'tick_volume' eksik, default değerle ekleniyor")
                data['tick_volume'] = 1
            
            data_with_indicators = self.data_processor.add_technical_indicators(data)
            
            if data_with_indicators is None:
                self.results['tests']['timeframes'][timeframe]['technical_indicators'] = {
                    'success': False,
                    'error': "Göstergeler hesaplanamadı"
                }
                return False
            
            # Performans metrikleri
            end_time = time.time()
            memory_used = self.get_memory_usage() - initial_memory
            time_taken = end_time - start_time
            
            # NaN kontrolü
            nan_columns = data_with_indicators.columns[data_with_indicators.isna().any()].tolist()
            
            self.results['tests']['timeframes'][timeframe]['technical_indicators'] = {
                'success': True,
                'indicators_added': len(data_with_indicators.columns) - len(data.columns),
                'total_columns': len(data_with_indicators.columns),
                'memory_mb': round(memory_used, 2),
                'time_seconds': round(time_taken, 2),
                'nan_columns': nan_columns
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Teknik gösterge testi sırasında hata: {str(e)}")
            self.results['tests']['timeframes'][timeframe]['technical_indicators'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_rl_state(self, data, timeframe):
        """RL state oluşturmayı test eder"""
        try:
            # prepare_rl_state metodu var mı kontrol et
            if not hasattr(self.data_processor, 'prepare_rl_state'):
                logger.warning("prepare_rl_state metodu henüz implement edilmemiş")
                self.results['tests']['timeframes'][timeframe]['rl_state'] = {
                    'success': False,
                    'warning': "prepare_rl_state metodu henüz implement edilmemiş",
                    'status': "not_implemented"
                }
                return False
            
            initial_memory = self.get_memory_usage()
            start_time = time.time()
            
            state = self.data_processor.prepare_rl_state(data.iloc[-1])
            
            if state is None:
                self.results['tests']['timeframes'][timeframe]['rl_state'] = {
                    'success': False,
                    'error': "RL state oluşturulamadı"
                }
                return False
            
            # Performans metrikleri
            end_time = time.time()
            memory_used = self.get_memory_usage() - initial_memory
            time_taken = end_time - start_time
            
            self.results['tests']['timeframes'][timeframe]['rl_state'] = {
                'success': True,
                'state_shape': state.shape if hasattr(state, 'shape') else None,
                'memory_mb': round(memory_used, 2),
                'time_seconds': round(time_taken, 2)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"RL state testi sırasında hata: {str(e)}")
            self.results['tests']['timeframes'][timeframe]['rl_state'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def run_all_tests(self):
        """Tüm testleri çalıştırır ve sonuçları kaydeder"""
        print("\n" + "="*50)
        print("           DEBUG TESTLERİ BAŞLATILIYOR           ")
        print("="*50 + "\n")

        # MT5 bağlantı testi
        print("\n[1] MT5 Bağlantı Testi")
        print("-"*30)
        self.test_mt5_connection()

        # Veri toplama konfigürasyonu
        data_collection_config = {
            "5m": 2000,
            "15m": 1000,
            "1h": 500
        }
        
        # Test sonuçları için dictionary'yi hazırla
        self.results['tests']['timeframes'] = {}
        for timeframe in data_collection_config.keys():
            self.results['tests']['timeframes'][timeframe] = {}
        
        all_tests_passed = True
        
        for timeframe, candles in data_collection_config.items():
            print(f"\n[2] {timeframe} Zaman Dilimi Testleri")
            print("-"*30)
            
            # Veri alımı testi
            print(f"\n• Veri alımı test ediliyor...")
            success, data = self.test_data_retrieval(timeframe, candles)
            print(f"  {'✓' if success else '✗'} Veri alımı")
            
            if success and data is not None:
                # Teknik gösterge testi
                print(f"\n• Teknik göstergeler test ediliyor...")
                success = self.test_technical_indicators(data, timeframe)
                print(f"  {'✓' if success else '✗'} Teknik göstergeler")
                
                # RL state testi
                print(f"\n• RL state test ediliyor...")
                success = self.test_rl_state(data, timeframe)
                print(f"  {'✓' if success else '✗'} RL state")
            
            all_tests_passed &= success

        # Sonuçları kaydet
        self.save_results()

        # Özet rapor
        print("\n" + "="*50)
        print("           DEBUG TESTLERİ TAMAMLANDI           ")
        print("="*50)
        self.print_summary()

        return self.results

    def save_results(self):
        """Test sonuçlarını kaydet"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=4)
            print(f"\nSonuçlar kaydedildi: {self.results_file}")
        except Exception as e:
            logger.error(f"Sonuçlar kaydedilirken hata: {str(e)}")

    def print_summary(self):
        """Test sonuçlarının özetini yazdır"""
        print("\nTest Sonuçları Özeti:")
        print("-"*30)
        
        # MT5 bağlantı sonucu
        mt5_success = self.results['tests']['mt5_connection'].get('success', False)
        print(f"\n• MT5 Bağlantısı: {'✓' if mt5_success else '✗'}")
        
        # Timeframe sonuçları
        for timeframe in self.results['tests']['timeframes']:
            print(f"\n• {timeframe} Testleri:")
            timeframe_results = self.results['tests']['timeframes'][timeframe]
            
            # Veri alımı
            data_success = timeframe_results.get('data_retrieval', {}).get('success', False)
            print(f"  {'✓' if data_success else '✗'} Veri Alımı")
            
            # Teknik göstergeler
            tech_success = timeframe_results.get('technical_indicators', {}).get('success', False)
            print(f"  {'✓' if tech_success else '✗'} Teknik Göstergeler")
            
            # RL state
            rl_success = timeframe_results.get('rl_state', {}).get('success', False)
            print(f"  {'✓' if rl_success else '✗'} RL State")
        
        print("\n" + "="*50)

def run_standalone_tests():
    """Bağımsız test fonksiyonu"""
    debugger = DataDebugger()
    return debugger.run_all_tests()

if __name__ == "__main__":
    # Loglama ayarlarını yapılandır
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('debug.log'),
            logging.StreamHandler()
        ]
    )
    
    # Test'i çalıştır
    run_standalone_tests() 