import pandas as pd
import numpy as np
import logging
from utils.mt5_connector import MT5Connector
from utils.data_processor import DataProcessor
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import traceback

# .env dosyasını yükle
load_dotenv()

# Logs klasörünü oluştur
if not os.path.exists('logs'):
    os.makedirs('logs')

# Logger ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/data_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataTest")

class DataTester:
    def __init__(self):
        """DataTester sınıfını başlat"""
        load_dotenv()
        
        # MT5 bağlantısı için parametreleri al
        mt5_login = int(os.getenv('MT5_LOGIN'))
        mt5_password = os.getenv('MT5_PASSWORD')
        mt5_server = os.getenv('MT5_SERVER')
        
        # MT5 bağlantısını oluştur
        self.mt5_connector = MT5Connector(
            login=mt5_login,
            password=mt5_password,
            server=mt5_server
        )
        
        # DataProcessor'ı başlat
        self.data_processor = DataProcessor(mt5_connector=self.mt5_connector)
        
        # Test sonuçları için sözlük yapısını oluştur
        self.test_results = {
            'data_retrieval': {},
            'data_processing': {},
            'feature_validation': {},
            'data_quality': {}
        }
    
    def test_data_retrieval(self, symbol="XAUUSD", timeframe="5m"):
        """Veri alımı testini gerçekleştir"""
        logger.info("Veri alımı testi başlatılıyor...")
        
        try:
            # Timeframe'e göre mum sayısını belirle
            num_candles = {
                "5m": 1000,
                "15m": 1000,
                "1h": 1000
            }.get(timeframe, 1000)
            
            # Veriyi al
            df = self.mt5_connector.get_historical_data(symbol, timeframe, num_candles=num_candles)
            
            if df is not None and len(df) > 0:
                self.test_results['data_retrieval'][timeframe] = {
                    'success': True,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'start_date': str(df.index[0]),
                    'end_date': str(df.index[-1])
                }
                logger.info(f"{timeframe} için {len(df)} satır veri alındı")
            else:
                raise ValueError(f"{timeframe} için veri alınamadı")
            
        except Exception as e:
            logger.error(f"Veri alımı testi hatası ({timeframe}): {str(e)}")
            self.test_results['data_retrieval'][timeframe] = {
                'success': False,
                'error': str(e)
            }
    
    def test_data_processing(self, symbol="XAUUSD", timeframe="5m", num_candles=1000):
        """Veri işleme sürecini test et"""
        logger.info("Veri işleme testi başlatılıyor...")
        
        try:
            # Ham veriyi al
            raw_df = self.mt5_connector.get_historical_data(symbol, timeframe, num_candles=num_candles)
            if raw_df is None:
                raise ValueError("Ham veri alınamadı")
            
            # Veri işleme adımlarını test et
            processing_steps = {
                'raw_data': {
                    'shape': raw_df.shape,
                    'columns': list(raw_df.columns)
                }
            }
            
            # 1. Veri kalitesi kontrolleri
            df_quality = self.data_processor._check_data_quality(raw_df.copy(), timeframe)
            processing_steps['quality_check'] = {
                'shape': df_quality.shape,
                'columns': list(df_quality.columns),
                'new_columns': list(set(df_quality.columns) - set(raw_df.columns))
            }
            
            # 2. İndikatör hesaplamaları
            df_indicators = self.data_processor._calculate_indicators(df_quality.copy())
            processing_steps['indicators'] = {
                'shape': df_indicators.shape,
                'columns': list(df_indicators.columns),
                'new_columns': list(set(df_indicators.columns) - set(df_quality.columns))
            }
            
            # 3. Seans yönetimi
            df_sessions = self.data_processor.enhance_session_management(df_indicators.copy())
            processing_steps['sessions'] = {
                'shape': df_sessions.shape,
                'columns': list(df_sessions.columns),
                'new_columns': list(set(df_sessions.columns) - set(df_indicators.columns))
            }
            
            # 4. Özellik normalizasyonu
            df_normalized = self.data_processor._normalize_features(df_sessions.copy())
            processing_steps['normalized'] = {
                'shape': df_normalized.shape,
                'columns': list(df_normalized.columns),
                'value_ranges': {
                    col: {
                        'min': df_normalized[col].min(),
                        'max': df_normalized[col].max(),
                        'mean': df_normalized[col].mean()
                    } for col in df_normalized.columns
                }
            }
            
            self.test_results['data_processing'] = processing_steps
            logger.info("Veri işleme testi tamamlandı")
            
        except Exception as e:
            logger.error(f"Veri işleme testi hatası: {str(e)}")
            self.test_results['data_processing'] = {
                'success': False,
                'error': str(e)
            }
    
    def test_feature_validation(self, symbol="XAUUSD", timeframe="5m", num_candles=1000):
        """Özellik doğrulamasını test et"""
        logger.info("Özellik doğrulama testi başlatılıyor...")
        
        try:
            # Veriyi al ve işle
            raw_df = self.mt5_connector.get_historical_data(symbol, timeframe, num_candles=num_candles)
            processed_df = self.data_processor.process_data(raw_df.copy(), timeframe)
            
            # Beklenen özellikleri kontrol et
            expected_features = (
                self.data_processor.price_features +
                self.data_processor.volume_features +
                self.data_processor.momentum_indicators +
                self.data_processor.trend_indicators +
                self.data_processor.volatility_indicators +
                self.data_processor.oscillator_indicators +
                self.data_processor.volume_indicators +
                self.data_processor.session_features +
                self.data_processor.quality_features
            )
            
            # Özellik kontrolü
            feature_validation = {
                'expected_features': expected_features,
                'actual_features': list(processed_df.columns),
                'missing_features': list(set(expected_features) - set(processed_df.columns)),
                'extra_features': list(set(processed_df.columns) - set(expected_features))
            }
            
            # Değer aralıklarını kontrol et
            value_ranges = {}
            for col in processed_df.columns:
                value_ranges[col] = {
                    'min': float(processed_df[col].min()),
                    'max': float(processed_df[col].max()),
                    'mean': float(processed_df[col].mean()),
                    'null_count': int(processed_df[col].isnull().sum()),
                    'inf_count': int(np.isinf(processed_df[col]).sum())
                }
            
            feature_validation['value_ranges'] = value_ranges
            self.test_results['feature_validation'] = feature_validation
            
            logger.info("Özellik doğrulama testi tamamlandı")
            
        except Exception as e:
            logger.error(f"Özellik doğrulama testi hatası: {str(e)}")
            self.test_results['feature_validation'] = {
                'success': False,
                'error': str(e)
            }
    
    def test_data_quality(self, symbol="XAUUSD", timeframe="5m", num_candles=1000):
        """Veri kalitesini test et"""
        logger.info("Veri kalitesi testi başlatılıyor...")
        
        try:
            # Ham veriyi al
            raw_df = self.mt5_connector.get_historical_data(symbol, timeframe, num_candles=num_candles)
            if raw_df is None:
                raise ValueError("Ham veri alınamadı")
            
            # Veriyi işle
            processed_df = self.data_processor.process_data(raw_df.copy(), timeframe)
            
            # Veri kalitesi kontrolü
            quality_checks = {
                'missing_values': processed_df.isnull().sum().to_dict(),
                'has_gaps': processed_df['has_gap'].sum(),
                'gap_sizes': {
                    'mean': float(processed_df['gap_size'].mean()),
                    'max': float(processed_df['gap_size'].max())
                },
                'outliers': processed_df['is_outlier'].sum(),
                'volume_spikes': processed_df['volume_spike'].sum(),
                'volatility': {
                    'mean': float(processed_df['volatility_factor'].mean()),
                    'std': float(processed_df['volatility_factor'].std())
                },
                'session_distribution': {
                    'asian': float(processed_df['asian_session'].mean()),
                    'london': float(processed_df['london_session'].mean()),
                    'ny': float(processed_df['ny_session'].mean())
                }
            }
            
            self.test_results['data_quality'][timeframe] = quality_checks
            
        except Exception as e:
            logger.error(f"Veri kalitesi testi hatası: {str(e)}")
            self.test_results['data_quality'][timeframe] = {
                'success': False,
                'error': str(e)
            }
    
    def run_all_tests(self, symbol="XAUUSD"):
        """Tüm testleri çalıştır ve sonuçları kaydet"""
        logger.info(f"{symbol} için tüm testler başlatılıyor...")
        
        timeframes = ["5m", "15m", "1h"]
        
        try:
            # Veri alma testleri
            for tf in timeframes:
                self.test_data_retrieval(symbol, tf)
            
            # Veri işleme testleri
            for tf in timeframes:
                self.test_data_processing(symbol, tf)
            
            # Özellik doğrulama testleri
            for tf in timeframes:
                self.test_feature_validation(symbol, tf)
            
            # Veri kalitesi testleri
            for tf in timeframes:
                self.test_data_quality(symbol, tf)
            
            # NumPy ve Pandas veri tiplerini Python'un temel veri tiplerine dönüştür
            def convert_to_basic_types(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                                  np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, (pd.Series, pd.DataFrame)):
                    return obj.to_dict()
                elif isinstance(obj, dict):
                    return {key: convert_to_basic_types(value) for key, value in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_basic_types(item) for item in obj]
                return obj
            
            # Test sonuçlarını dönüştür
            converted_results = convert_to_basic_types(self.test_results)
            
            # Sonuçları JSON olarak kaydet
            with open('test_results.json', 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=4, ensure_ascii=False)
            
            logger.info("Test sonuçları başarıyla kaydedildi: test_results.json")
            
            # Test sonuçlarını logla
            logger.info("\n=== Test Sonuçları ===")
            
            # Veri alımı sonuçları
            logger.info("\n1. Veri Alımı Testleri:")
            for timeframe, result in converted_results['data_retrieval'].items():
                if result.get('success', False):
                    logger.info(f"✓ {timeframe}: {result['rows']} satır veri alındı")
                else:
                    logger.error(f"✗ {timeframe}: {result.get('error', 'Bilinmeyen hata')}")
            
            # Veri işleme sonuçları
            logger.info("\n2. Veri İşleme Testleri:")
            if 'success' in converted_results['data_processing']:
                if not converted_results['data_processing']['success']:
                    logger.error(f"✗ Hata: {converted_results['data_processing']['error']}")
                else:
                    for step, info in converted_results['data_processing'].items():
                        if step != 'success':
                            logger.info(f"✓ {step}: {info['shape'][0]} satır, {info['shape'][1]} sütun")
            
            # Özellik doğrulama sonuçları
            logger.info("\n3. Özellik Doğrulama Testleri:")
            if 'success' in converted_results['feature_validation']:
                if not converted_results['feature_validation']['success']:
                    logger.error(f"✗ Hata: {converted_results['feature_validation']['error']}")
                else:
                    missing = converted_results['feature_validation']['missing_features']
                    extra = converted_results['feature_validation']['extra_features']
                    logger.info(f"{'✓' if not missing else '✗'} Eksik özellikler: {missing if missing else 'Yok'}")
                    logger.info(f"{'✓' if not extra else '!'} Fazla özellikler: {extra if extra else 'Yok'}")
            
            # Veri kalitesi sonuçları
            logger.info("\n4. Veri Kalitesi Testleri:")
            for timeframe, results in converted_results['data_quality'].items():
                logger.info(f"\n{timeframe} sonuçları:")
                if 'success' in results and not results['success']:
                    logger.error(f"✗ Hata: {results['error']}")
                else:
                    logger.info(f"✓ Boşluklar: {results['has_gaps']} adet")
                    logger.info(f"✓ Aykırı değerler: {results['outliers']} adet")
                    logger.info(f"✓ Hacim spike'ları: {results['volume_spikes']} adet")
                    logger.info(f"✓ Volatilite (ortalama): {results['volatility']['mean']:.4f}")
                    logger.info(f"✓ Seans dağılımı:")
                    logger.info(f"  - Asya: {results['session_distribution']['asian']*100:.1f}%")
                    logger.info(f"  - Londra: {results['session_distribution']['london']*100:.1f}%")
                    logger.info(f"  - NY: {results['session_distribution']['ny']*100:.1f}%")
            
            logger.info("\nTest tamamlandı!")
            
        except Exception as e:
            logger.error(f"Test çalıştırma hatası: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        # Test sınıfını başlat
        tester = DataTester()
        
        logger.info("=== XAUUSD Veri Test Sonuçları ===")
        
        # Tüm testleri çalıştır
        tester.run_all_tests()
        
        # Sonuçları logla
        logger.info("\n=== Test Sonuçları ===")
        
        # Veri alımı sonuçları
        logger.info("\n1. Veri Alımı Testleri:")
        for timeframe, result in tester.test_results['data_retrieval'].items():
            if result.get('success', False):
                logger.info(f"✓ {timeframe}: {result['rows']} satır veri alındı")
            else:
                logger.error(f"✗ {timeframe}: {result.get('error', 'Bilinmeyen hata')}")
        
        # Veri işleme sonuçları
        logger.info("\n2. Veri İşleme Testleri:")
        if 'success' in tester.test_results['data_processing']:
            if not tester.test_results['data_processing']['success']:
                logger.error(f"✗ Hata: {tester.test_results['data_processing']['error']}")
            else:
                for step, info in tester.test_results['data_processing'].items():
                    if step != 'success':
                        logger.info(f"✓ {step}: {info['shape'][0]} satır, {info['shape'][1]} sütun")
        
        # Özellik doğrulama sonuçları
        logger.info("\n3. Özellik Doğrulama Testleri:")
        if 'success' in tester.test_results['feature_validation']:
            if not tester.test_results['feature_validation']['success']:
                logger.error(f"✗ Hata: {tester.test_results['feature_validation']['error']}")
            else:
                missing = tester.test_results['feature_validation']['missing_features']
                extra = tester.test_results['feature_validation']['extra_features']
                logger.info(f"{'✓' if not missing else '✗'} Eksik özellikler: {missing if missing else 'Yok'}")
                logger.info(f"{'✓' if not extra else '!'} Fazla özellikler: {extra if extra else 'Yok'}")
        
        # Veri kalitesi sonuçları
        logger.info("\n4. Veri Kalitesi Testleri:")
        for timeframe, results in tester.test_results['data_quality'].items():
            logger.info(f"\n{timeframe} sonuçları:")
            if 'success' in results and not results['success']:
                logger.error(f"✗ Hata: {results['error']}")
            else:
                logger.info(f"✓ Boşluklar: {results['has_gaps']} adet")
                logger.info(f"✓ Aykırı değerler: {results['outliers']} adet")
                logger.info(f"✓ Hacim spike'ları: {results['volume_spikes']} adet")
                logger.info(f"✓ Volatilite (ortalama): {results['volatility']['mean']:.4f}")
                logger.info(f"✓ Seans dağılımı:")
                logger.info(f"  - Asya: {results['session_distribution']['asian']*100:.1f}%")
                logger.info(f"  - Londra: {results['session_distribution']['london']*100:.1f}%")
                logger.info(f"  - NY: {results['session_distribution']['ny']*100:.1f}%")
        
        logger.info("\nTest tamamlandı!")
        
    except Exception as e:
        logger.error(f"\n❌ Test sırasında hata oluştu: {str(e)}")
        logger.error("\nHata detayları:")
        logger.error(traceback.format_exc()) 