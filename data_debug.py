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
        
    def get_memory_usage(self):
        """Mevcut bellek kullanımını MB cinsinden döndürür"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_mt5_connection(self):
        """MT5 bağlantısını test eder"""
        try:
            if not self.mt5.connected and not self.mt5.connect():
                logger.error("MT5 bağlantısı başarısız!")
                return False, "Bağlantı kurulamadı"
                
            account_info = self.mt5.get_account_info()
            if not account_info:
                return False, "Hesap bilgileri alınamadı"
            
            logger.info(f"MT5 bağlantısı başarılı. Hesap: {account_info.login}")
            return True, {
                "login": account_info.login,
                "server": account_info.server,
                "balance": account_info.balance,
                "equity": account_info.equity
            }
        except Exception as e:
            logger.error(f"MT5 bağlantı testi sırasında hata: {str(e)}")
            return False, str(e)
    
    def test_data_retrieval(self, timeframe, num_candles):
        """Belirli bir zaman dilimi için veri alımını test eder"""
        try:
            initial_memory = self.get_memory_usage()
            start_time = time.time()
            
            data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
            
            if data is None:
                return False, f"Veri alınamadı - {timeframe}"
            
            # Veri doğrulama
            required_columns = ['open', 'high', 'low', 'close', 'time']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                return False, f"Eksik sütunlar: {missing_columns}"
            
            # Performans metrikleri
            end_time = time.time()
            memory_used = self.get_memory_usage() - initial_memory
            time_taken = end_time - start_time
            
            return True, {
                "rows": len(data),
                "columns": list(data.columns),
                "memory_mb": round(memory_used, 2),
                "time_seconds": round(time_taken, 2),
                "first_date": data['time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                "last_date": data['time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Veri alımı testi sırasında hata: {str(e)}")
            return False, str(e)
    
    def test_technical_indicators(self, data):
        """Teknik göstergelerin hesaplanmasını test eder"""
        try:
            initial_memory = self.get_memory_usage()
            start_time = time.time()
            
            if 'tick_volume' not in data.columns:
                logger.warning("'tick_volume' eksik, default değerle ekleniyor")
                data['tick_volume'] = 1
            
            data_with_indicators = self.data_processor.add_technical_indicators(data)
            
            if data_with_indicators is None:
                return False, "Göstergeler hesaplanamadı"
            
            # Performans metrikleri
            end_time = time.time()
            memory_used = self.get_memory_usage() - initial_memory
            time_taken = end_time - start_time
            
            # NaN kontrolü
            nan_columns = data_with_indicators.columns[data_with_indicators.isna().any()].tolist()
            
            return True, {
                "indicators_added": len(data_with_indicators.columns) - len(data.columns),
                "total_columns": len(data_with_indicators.columns),
                "memory_mb": round(memory_used, 2),
                "time_seconds": round(time_taken, 2),
                "nan_columns": nan_columns
            }
            
        except Exception as e:
            logger.error(f"Teknik gösterge testi sırasında hata: {str(e)}")
            return False, str(e)
    
    def test_rl_state(self, data):
        """RL state oluşturmayı test eder"""
        try:
            # prepare_rl_state metodu var mı kontrol et
            if not hasattr(self.data_processor, 'prepare_rl_state'):
                logger.warning("prepare_rl_state metodu henüz implement edilmemiş")
                return True, {
                    "warning": "prepare_rl_state metodu henüz implement edilmemiş",
                    "status": "not_implemented"
                }
            
            initial_memory = self.get_memory_usage()
            start_time = time.time()
            
            state = self.data_processor.prepare_rl_state(data.iloc[-1])
            
            if state is None:
                return False, "RL state oluşturulamadı"
            
            # Performans metrikleri
            end_time = time.time()
            memory_used = self.get_memory_usage() - initial_memory
            time_taken = end_time - start_time
            
            return True, {
                "state_shape": state.shape if hasattr(state, 'shape') else None,
                "memory_mb": round(memory_used, 2),
                "time_seconds": round(time_taken, 2)
            }
            
        except Exception as e:
            logger.error(f"RL state testi sırasında hata: {str(e)}")
            return True, {
                "warning": str(e),
                "status": "error"
            }
    
    def run_all_tests(self):
        """Tüm testleri çalıştırır ve sonuçları kaydeder"""
        print("\n==== Debug Testleri Başlatılıyor ====")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "memory_total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
                "memory_available": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
                "cpu_percent": psutil.cpu_percent()
            },
            "tests": {
                "mt5_connection": {"success": False, "result": None},  # Initialize with default values
                "timeframes": {}
            }
        }
        
        # MT5 bağlantı testi
        print("\nMT5 bağlantısı test ediliyor...")
        success, connection_result = self.test_mt5_connection()
        results["tests"]["mt5_connection"] = {
            "success": success,
            "result": connection_result
        }
        print(f"MT5 Bağlantısı: {'✅' if success else '❌'}")
        if success and isinstance(connection_result, dict):
            print(f"  Hesap: {connection_result['login']}")
            print(f"  Bakiye: ${connection_result['balance']:.2f}")
        
        # Her timeframe için testler
        timeframes = {
            "1m": 5000,
            "5m": 2000,
            "15m": 1000
        }
        
        all_tests_passed = True
        
        for timeframe, candles in timeframes.items():
            print(f"\n{timeframe} zaman dilimi test ediliyor...")
            results["tests"]["timeframes"][timeframe] = {}
            
            # Veri alımı testi
            print(f"  Veri alımı test ediliyor...")
            success, data_result = self.test_data_retrieval(timeframe, candles)
            results["tests"]["timeframes"][timeframe]["data_retrieval"] = {
                "success": success,
                "result": data_result
            }
            print(f"  Veri Alımı: {'✅' if success else '❌'}")
            all_tests_passed &= success
            
            if success:
                data = self.mt5.get_historical_data("XAUUSD", timeframe, num_candles=candles)
                
                # Teknik gösterge testi
                print(f"  Teknik göstergeler test ediliyor...")
                success, indicator_result = self.test_technical_indicators(data)
                results["tests"]["timeframes"][timeframe]["technical_indicators"] = {
                    "success": success,
                    "result": indicator_result
                }
                print(f"  Teknik Göstergeler: {'✅' if success else '❌'}")
                all_tests_passed &= success
                
                # RL state testi
                print(f"  RL state test ediliyor...")
                success, state_result = self.test_rl_state(data)
                results["tests"]["timeframes"][timeframe]["rl_state"] = {
                    "success": success,
                    "result": state_result
                }
                status = state_result.get('status') if isinstance(state_result, dict) else None
                if status == 'not_implemented':
                    print(f"  RL State: ⚠️ (Henüz implement edilmemiş)")
                elif status == 'error':
                    print(f"  RL State: ⚠️ ({state_result.get('warning', 'Bilinmeyen hata')})")
                else:
                    print(f"  RL State: {'✅' if success else '❌'}")
        
        # Sonuçları kaydet
        try:
            with open(self.results_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nSonuçlar kaydedildi: {self.results_file}")
        except Exception as e:
            logger.error(f"Sonuçlar kaydedilirken hata: {str(e)}")
        
        print("\n==== Debug Testleri Tamamlandı ====")
        
        # Test sonuçlarını özetle
        print("\nTest Sonuçları Özeti:")
        print(f"MT5 Bağlantısı: {'✅' if results['tests']['mt5_connection']['success'] else '❌'}")
        for timeframe in timeframes:
            print(f"\n{timeframe} Testleri:")
            tf_results = results["tests"]["timeframes"][timeframe]
            print(f"  Veri Alımı: {'✅' if tf_results['data_retrieval']['success'] else '❌'}")
            if 'technical_indicators' in tf_results:
                print(f"  Teknik Göstergeler: {'✅' if tf_results['technical_indicators']['success'] else '❌'}")
            if 'rl_state' in tf_results:
                state_result = tf_results['rl_state']['result']
                if isinstance(state_result, dict):
                    status = state_result.get('status')
                    if status == 'not_implemented':
                        print(f"  RL State: ⚠️ (Henüz implement edilmemiş)")
                    elif status == 'error':
                        print(f"  RL State: ⚠️ ({state_result.get('warning', 'Bilinmeyen hata')})")
                    else:
                        print(f"  RL State: {'✅' if tf_results['rl_state']['success'] else '❌'}")
        
        return results

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