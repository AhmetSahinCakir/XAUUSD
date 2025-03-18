import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import logging
import ta
import pytz
from datetime import datetime, time
from functools import lru_cache

logger = logging.getLogger("TradingBot.DataProcessor")

# Seans zaman dilimlerini tanımla (UTC)
ASIA_SESSION = (time(0, 0), time(9, 0))
EUROPE_SESSION = (time(9, 0), time(17, 0))
US_SESSION = (time(17, 0), time(23, 59))

class DataProcessor:
    def __init__(self, mt5_connector=None):
        """
        Veri işleme sınıfı başlatıcı
        """
        self.logger = logging.getLogger('TradingBot.DataProcessor')
        
        # MT5 bağlantısı
        if mt5_connector is None:
            from utils.mt5_connector import MT5Connector
            from config.config import MT5_CONFIG
            self.mt5_connector = MT5Connector(
                login=MT5_CONFIG['login'],
                password=MT5_CONFIG['password'],
                server=MT5_CONFIG['server']
            )
            if not self.mt5_connector.connect():
                logger.error("MT5 bağlantısı başlatılamadı!")
        else:
            self.mt5_connector = mt5_connector
        
        # Temel özellik grupları
        self.price_features = ['open', 'high', 'low', 'close']
        self.volume_features = ['tick_volume']
        self.technical_indicators = [
            'rsi', 'macd', 'macd_signal', 'atr',
            'bb_upper', 'bb_lower', 'sma_20'
        ]
        self.session_features = ['asian_session', 'london_session', 'ny_session']
        self.quality_features = ['has_gap', 'gap_size']
        
        # Tüm özellikleri birleştir
        self.all_features = (
            self.price_features +
            self.volume_features +
            self.technical_indicators +
            self.quality_features +
            self.session_features
        )
        
        # Özellik sayısı
        self.n_features = len(self.all_features)
        logger.info(f"DataProcessor başlatıldı. Toplam özellik sayısı: {self.n_features}")
        
        # Scaler
        self.feature_scaler = MinMaxScaler()
        self.feature_scaler_fitted = False
        
        # Cache için LRU
        self.cache_size = 128
        
        # Veri kalitesi parametreleri
        self.quality_params = {
            'outlier_std': 3.0,
            'volume_spike_threshold': 5.0,
            'gap_threshold_minutes': {
                '5m': 10,
                '15m': 30,
                '1h': 120
            },
            'volatility_window': 20,
            'session_buffer_minutes': 30
        }
        
        # Veri alımı parametreleri
        self.lookback_periods = {
            '5m': 10000,
            '15m': 5000,
            '1h': 2000
        }
        
        # Özellik isimleri ve grupları
        self._initialize_feature_groups()
        
        # Scaler'ları başlat
        self._initialize_scalers()
        
        # Veri çerçevesi
        self.df = None
        
        # Özellik grupları
        self.momentum_indicators = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'mfi', 'roc']
        self.trend_indicators = ['sma_10', 'sma_50', 'ema_10', 'ema_50', 'adx']
        self.volatility_indicators = ['atr', 'bb_upper', 'bb_middle', 'bb_lower']
        self.oscillator_indicators = ['stoch_k', 'stoch_d', 'williams_r', 'cci']
        self.volume_indicators = ['obv']
        self.quality_features = ['has_gap', 'gap_size', 'is_outlier', 'volume_spike', 'volatility_factor']
        
        # Scaler'lar
        self.feature_scaler = MinMaxScaler()
        self.feature_scaler_fitted = False
        
        self.feature_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        self.all_features = [
            'open', 'high', 'low', 'close', 'tick_volume',  # 5 price features
            'rsi', 'macd', 'macd_signal', 'atr', 'bb_upper', 'bb_lower', 'sma_20',  # 7 technical indicators
            'has_gap', 'gap_size', 'asian_session', 'london_session', 'ny_session'  # 5 yeni özellik
        ]  # Total 17 features + 3 account state = 20 features
        
        # Cache for technical indicators
        self.indicators_cache = {}
        self.cache_max_size = 10  # Maximum number of DataFrames to keep in cache
        
    def _initialize_feature_groups(self):
        """Özellik gruplarını ve isimlerini başlatır"""
        self.price_features = ['open', 'high', 'low', 'close']
        self.volume_features = ['volume', 'tick_volume']
        self.momentum_indicators = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'mfi', 'roc']
        self.trend_indicators = ['sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'ema_50', 'adx']
        self.volatility_indicators = ['atr', 'bb_upper', 'bb_middle', 'bb_lower']
        self.oscillator_indicators = ['stoch_k', 'stoch_d', 'williams_r', 'cci']
        self.volume_indicators = ['obv']
        self.session_features = ['session_asian', 'session_london', 'session_ny']
        self.quality_features = ['has_gap', 'gap_size', 'is_outlier', 'volume_spike', 'volatility_factor']
        
        # Tüm özellikleri birleştir
        self.feature_names = (
            self.price_features +
            self.volume_features +
            self.momentum_indicators +
            self.trend_indicators +
            self.volatility_indicators +
            self.oscillator_indicators +
            self.volume_indicators +
            self.session_features +
            self.quality_features
        )
        
        self.n_features = len(self.feature_names)

    def _initialize_scalers(self):
        """Scaler'ları başlatır"""
        self.scalers = {
            'price': MinMaxScaler(),
            'volume': MinMaxScaler(),
            'momentum': MinMaxScaler(),
            'trend': MinMaxScaler(),
            'volatility': MinMaxScaler(),
            'oscillator': MinMaxScaler(),
            'quality': MinMaxScaler()
        }
        self.scalers_fitted = {k: False for k in self.scalers.keys()}

    @lru_cache(maxsize=128)
    def calculate_adaptive_periods(self, volatility):
        """Volatiliteye göre adaptif periyotlar hesaplar"""
        base_periods = {
            'rsi': 14,
            'macd': (12, 26, 9),  # (fast, slow, signal)
            'bb': 20,
            'stoch': 14,
            'atr': 14
        }
        
        # Volatilite faktörü (0.5 ile 1.5 arasında)
        vol_factor = 1 + (volatility - 0.02) / 0.04  # 0.02 normal volatilite kabul edilir
        vol_factor = max(0.5, min(1.5, vol_factor))
        
        return {
            'rsi': max(5, int(base_periods['rsi'] * vol_factor)),
            'macd': tuple(max(5, int(p * vol_factor)) for p in base_periods['macd']),
            'bb': max(5, int(base_periods['bb'] * vol_factor)),
            'stoch': max(5, int(base_periods['stoch'] * vol_factor)),
            'atr': max(5, int(base_periods['atr'] * vol_factor))
        }

    def detect_outliers(self, df):
        """Aykırı değerleri tespit et"""
        try:
            # Fiyat ve hacim sütunları için aykırı değerleri kontrol et
            columns = ['open', 'high', 'low', 'close', 'tick_volume']
            
            # Her sütun için z-score hesapla
            for col in columns:
                if col in df.columns:
                    z_score = (df[col] - df[col].mean()) / df[col].std()
                    df[f'{col}_is_outlier'] = abs(z_score) > 3
            
            # Herhangi bir sütunda aykırı değer varsa işaretle
            df['is_outlier'] = df[[f'{col}_is_outlier' for col in columns if f'{col}_is_outlier' in df.columns]].any(axis=1)
            
            # Geçici sütunları temizle
            for col in columns:
                if f'{col}_is_outlier' in df.columns:
                    df = df.drop(f'{col}_is_outlier', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Aykırı değer tespiti hatası: {str(e)}")
            return df

    def detect_volume_spikes(self, df):
        """Hacim sıçramalarını tespit et"""
        try:
            # Hacim ortalaması ve standart sapması
            volume_mean = df['tick_volume'].rolling(window=20).mean()
            volume_std = df['tick_volume'].rolling(window=20).std()
            
            # Hacim spike'larını tespit et (3 standart sapma üzeri)
            df['volume_spike'] = (df['tick_volume'] > (volume_mean + 3 * volume_std))
            
            return df
            
        except Exception as e:
            logger.error(f"Hacim spike tespiti hatası: {str(e)}")
            return df

    def calculate_volatility_factor(self, df):
        """Volatilite faktörü hesaplar"""
        returns = df['close'].pct_change()
        volatility = returns.rolling(
            window=self.quality_params['volatility_window'],
            min_periods=1
        ).std()
        
        # Volatilite faktörünü 0.5 ile 2 arasında normalize et
        normalized_vol = 0.5 + (volatility - volatility.min()) / (volatility.max() - volatility.min()) * 1.5
        return normalized_vol

    def enhance_session_management(self, df):
        """Seans bilgilerini ekle ve yönet"""
        try:
            # Zaman sütununu UTC'ye dönüştür
            if not df.index.tz:
                df.index = df.index.tz_localize('UTC')
            
            # Her satır için saat bilgisini al (UTC)
            hours = df.index.hour
            minutes = df.index.minute
            
            # Seans tanımları (UTC)
            df['asian_session'] = ((hours >= 0) & (hours < 8)).astype(int)
            df['london_session'] = ((hours >= 8) & (hours < 16)).astype(int)
            df['ny_session'] = ((hours >= 13) & (hours < 21)).astype(int)
            
            # Seans geçişlerinde tampon süre ekle
            buffer_minutes = self.quality_params['session_buffer_minutes']
            
            # Tampon süreleri ekle
            df.loc[(minutes < buffer_minutes) & (df['asian_session'] == 1), 'asian_session'] = 0
            df.loc[(minutes < buffer_minutes) & (df['london_session'] == 1), 'london_session'] = 0
            df.loc[(minutes < buffer_minutes) & (df['ny_session'] == 1), 'ny_session'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Seans yönetimi hatası: {str(e)}")
            return df

    def process_data(self, df, timeframe):
        """Veriyi işle ve özellikleri hesapla"""
        try:
            # Veri kontrolü
            if df is None or len(df) == 0:
                logger.error("İşlenecek veri bulunamadı veya boş")
                return None
            
            # NaN değerleri temizle
            df = df.copy()
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Temel özellikler
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Teknik göstergeler
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bantları
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['sma_20'] = bb.bollinger_mavg()
            
            # ATR
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Fiyat boşluklarını tespit et
            df = self.detect_price_gaps(df, timeframe)
            
            # Seans bilgilerini ekle
            df = self.enhance_session_management(df)
            
            # Hedef değişkeni hesapla
            df['target'] = df['close'].pct_change(periods=1).shift(-1)
            df['target'] = df['target'].fillna(0)
            df['target'] = df['target'].clip(-0.1, 0.1)
            
            # NaN değerleri temizle
            df = df.dropna()
            
            # Özellik sütunlarını kontrol et
            missing_features = set(self.all_features) - set(df.columns)
            if missing_features:
                logger.warning(f"Eksik özellikler: {missing_features}")
                for feature in missing_features:
                    df[feature] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Veri işleme hatası: {str(e)}")
            return None

    def _check_data_quality(self, df, timeframe):
        """Veri kalitesi kontrollerini gerçekleştir"""
        try:
            # Boşlukları tespit et
            df = self.detect_price_gaps(df, timeframe)
            
            # Aykırı değerleri tespit et
            df = self.detect_outliers(df)
            
            # Hacim spike'larını tespit et
            df = self.detect_volume_spikes(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Veri kalitesi kontrolü hatası: {str(e)}")
            raise

    def _calculate_indicators(self, df):
        """Teknik indikatörleri hesaplar"""
        # Volatiliteye göre adaptif periyotları hesapla
        volatility = df['close'].pct_change().std()
        periods = self.calculate_adaptive_periods(volatility)
        
        # Momentum indikatörleri
        df['rsi'] = ta.momentum.rsi(df['close'], window=periods['rsi'])
        
        # MACD hesaplama - güncellenmiş versiyon
        macd_fast = ta.trend.ema_indicator(df['close'], window=periods['macd'][0])
        macd_slow = ta.trend.ema_indicator(df['close'], window=periods['macd'][1])
        df['macd'] = macd_fast - macd_slow
        df['macd_signal'] = ta.trend.ema_indicator(df['macd'], window=periods['macd'][2])
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Trend indikatörleri
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # Volatilite indikatörleri
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'],
            window=periods['atr']
        )
        
        bollinger = ta.volatility.BollingerBands(
            df['close'],
            window=periods['bb'],
            window_dev=2
        )
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # Osilatörler
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'],
            window=periods['stoch']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        df['williams_r'] = ta.momentum.williams_r(
            df['high'], df['low'], df['close'],
            lbp=periods['stoch']
        )
        
        # NaN değerleri doldur
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df

    def _normalize_features(self, df):
        """Özellikleri normalize eder"""
        feature_groups = {
            'price': self.price_features,
            'volume': self.volume_features,
            'momentum': self.momentum_indicators,
            'trend': self.trend_indicators,
            'volatility': self.volatility_indicators,
            'oscillator': self.oscillator_indicators,
            'quality': self.quality_features
        }
        
        for group, features in feature_groups.items():
            valid_features = [f for f in features if f in df.columns]
            if not valid_features:
                continue
                
            if not self.scalers_fitted[group]:
                self.scalers[group].fit(df[valid_features])
                self.scalers_fitted[group] = True
            
            df[valid_features] = self.scalers[group].transform(df[valid_features])
        
        return df

    def detect_price_gaps(self, df, timeframe):
        """Fiyat boşluklarını tespit et"""
        try:
            # Zaman farkını dakika cinsinden hesapla
            df['time_diff'] = df.index.to_series().diff().dt.total_seconds() / 60
            
            # Timeframe'e göre beklenen zaman farkı
            expected_diff = {
                '5m': 5,
                '15m': 15,
                '1h': 60
            }.get(timeframe, 5)
            
            # Boşlukları tespit et
            df['has_gap'] = df['time_diff'] > expected_diff * 1.5
            
            # Boşluk büyüklüğünü hesapla (ATR'ye göre normalize edilmiş)
            atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['gap_size'] = np.where(df['has_gap'],
                                     abs(df['open'] - df['close'].shift(1)) / atr,
                                     0)
            
            return df
            
        except Exception as e:
            logger.error(f"Fiyat boşluğu tespiti hatası: {str(e)}")
            return df

    def calculate_gap_size(self, df):
        """
        Fiyat boşluklarının büyüklüğünü hesaplar
        
        Parametreler:
        - df: İşlenecek DataFrame
        
        Dönüş:
        - Gap büyüklüğü serisi
        """
        try:
            if 'ATR' not in df.columns:
                df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
                df['ATR'] = df['ATR'].fillna(method='ffill').fillna(method='bfill')
            
            # Fiyat farkını ATR'ye göre normalize et
            gap_size = abs(df['open'] - df['close'].shift(1)) / df['ATR']
            
            # Sonsuz ve NaN değerleri makul bir maksimum değere değiştir
            max_allowed_gap = 10.0
            gap_size = gap_size.replace([np.inf, -np.inf], max_allowed_gap)
            gap_size = gap_size.fillna(0)
            
            # Gap olmayan yerlerde büyüklük 0 olmalı
            gap_size = gap_size * df['has_gap']
            
            return gap_size
        
        except Exception as e:
            self.logger.error(f"Gap büyüklüğü hesaplanırken hata: {str(e)}")
            return pd.Series(0, index=df.index)

    def add_session_info(self, df):
        """
        Mum verilerine seans bilgilerini ekler (Asya, Avrupa, ABD)
        
        Parametreler:
        - df: İşlenecek DataFrame
        
        Dönüş:
        - Seans bilgileri eklenmiş DataFrame
        """
        try:
            if 'time' not in df.columns:
                self.logger.debug("Zaman sütunu bulunamadı, seans bilgileri eklenemedi")
                df['session_asian'] = 0
                df['session_london'] = 0
                df['session_ny'] = 0
                return df
            
            df = df.copy()
            
            # Zaman sütununu UTC'ye çevir
            if df['time'].dt.tz is None:
                df['time_utc'] = df['time'].dt.tz_localize('UTC')
            else:
                df['time_utc'] = df['time'].dt.tz_convert('UTC')
            
            # Saat bilgisini çıkar
            df['hour'] = df['time_utc'].dt.hour
            df['minute'] = df['time_utc'].dt.minute
            
            # Seans bilgilerini ekle
            df['session_asian'] = df.apply(lambda x: 1 if ASIA_SESSION[0] <= time(x['hour'], x['minute']) < ASIA_SESSION[1] else 0, axis=1)
            df['session_london'] = df.apply(lambda x: 1 if EUROPE_SESSION[0] <= time(x['hour'], x['minute']) < EUROPE_SESSION[1] else 0, axis=1)
            df['session_ny'] = df.apply(lambda x: 1 if US_SESSION[0] <= time(x['hour'], x['minute']) <= US_SESSION[1] else 0, axis=1)
            
            # Seans dağılımını logla (sadece debug seviyesinde)
            self.logger.debug(f"Seans dağılımı: Asya: {df['session_asian'].mean()*100:.1f}%, "
                       f"Avrupa: {df['session_london'].mean()*100:.1f}%, "
                       f"ABD: {df['session_ny'].mean()*100:.1f}%")
            
            # Geçici sütunları kaldır
            df = df.drop(['time_utc', 'hour', 'minute'], axis=1, errors='ignore')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Seans bilgileri eklenirken hata: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Hata durumunda varsayılan değerlerle devam et
            df['session_asian'] = 0
            df['session_london'] = 0
            df['session_ny'] = 0
            return df
    
    def add_technical_indicators(self, df, timeframe):
        """
        Teknik göstergeleri hesaplar ve DataFrame'e ekler
        
        Args:
            df: İşlenecek DataFrame
            timeframe: Veri zaman aralığı ('5m', '15m', '1h', etc.)
        
        Returns:
            DataFrame: Teknik göstergeler eklenmiş DataFrame
        """
        try:
            # Önce hedef değişkeni hesapla (ham veriden)
            df['target'] = df['close'].pct_change(periods=1).shift(-1)
            df['target'] = df['target'].fillna(0)  # Son satırlar için NaN değerleri 0 ile doldur
            df['target'] = df['target'].clip(-0.1, 0.1)  # ±%10 ile sınırla
            
            # Teknik göstergeleri ekle
            df = self._calculate_indicators(df)
            if df is None:
                return None
            
            # NaN değerleri temizle
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Tüm özelliklerin var olduğunu kontrol et
            df = self.all_features_exist(df)
            if df is None:
                return None
            
            # Seans bilgilerini ekle
            df = self.add_session_info(df)
            
            # NaN değerleri tespit et ve raporla
            nan_cols = [col for col in self.all_features if col in df.columns and df[col].isnull().any()]
            if nan_cols:
                nan_counts = {col: df[col].isnull().sum() for col in nan_cols}
                logger.warning(f"Teknik göstergelerde NaN değerler var: {nan_cols}")
                logger.warning(f"NaN değer sayıları: {nan_counts}")
                
                # NaN değerleri temizle - önce forward fill, sonra backward fill
                for col in nan_cols:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            return df
            
        except Exception as e:
            logger.error(f"Teknik göstergeler hesaplanırken hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def all_features_exist(self, df):
        """DataFrame'de eksik sütunları kontrol eder ve eksikleri tamamlar
        
        Parametreler:
        - df: Kontrol edilecek DataFrame
        
        Dönüş:
        - Eksik sütunlar tamamlanmış DataFrame
        """
        if df is None:
            return None
        
        # all_features listesinde olup DataFrame'de olmayan sütunları bul
        missing_cols = [col for col in self.all_features if col not in df.columns]
        
        # Eksik sütunları varsayılan değerlerle doldur
        if missing_cols:
            logger.warning(f"DataFrame'de eksik sütunlar bulundu: {missing_cols}")
            
            default_values = {
                'rsi': 50,           # Nötr
                'macd': 0,           # Nötr
                'macd_signal': 0,    # Nötr
                'atr': 0.0001,       # Küçük bir değer
                'bb_upper': df['close'].mean() * 1.02 if 'close' in df.columns else 1,  # Ortalama +%2
                'bb_lower': df['close'].mean() * 0.98 if 'close' in df.columns else 1,  # Ortalama -%2
                'sma_20': df['close'].mean() if 'close' in df.columns else 1,  # Ortalama
                'has_gap': 0,        # Gap yok
                'gap_size': 0,        # Gap boyutu
                'asian_session': 0,   # Seans bilgileri
                'london_session': 0,  # Seans bilgileri
                'ny_session': 0        # Seans bilgileri
            }
            
            for col in missing_cols:
                if col in default_values:
                    df[col] = default_values[col]
                else:
                    df[col] = 0  # Diğer bilinmeyen sütunlar için 0
                logger.debug(f"Eksik sütun eklendi: {col}")
        
        return df

    def get_feature_names(self):
        """Özellik isimlerini döndürür"""
        return self.feature_names.copy()

    def process_training_data(self, data_dict):
        """
        Farklı zaman dilimleri için alınan verileri işler ve eğitim için hazırlar
        
        Args:
            data_dict: Dict, her zaman dilimi için DataFrame içeren sözlük
        
        Returns:
            DataFrame: İşlenmiş ve birleştirilmiş veri
        """
        try:
            processed_data = {}
            
            # Her zaman dilimi için veriyi işle
            for timeframe, df in data_dict.items():
                logger.info(f"{timeframe} verisi işleniyor...")
                
                # Veri kalitesi kontrolü
                if df.isnull().values.any():
                    missing_count = df.isnull().sum().sum()
                    logger.warning(f"{timeframe} için {missing_count} eksik veri bulundu. Dolduruluyor...")
                    df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Teknik göstergeleri ekle
                logger.info(f"{timeframe} için teknik göstergeler ekleniyor...")
                df_with_indicators = self.add_technical_indicators(df.copy(), timeframe)
                if df_with_indicators is None:
                    logger.error(f"{timeframe} için teknik göstergeler eklenemedi")
                    continue
                
                # Fiyat boşluklarını tespit et
                logger.info(f"{timeframe} için fiyat boşlukları tespit ediliyor...")
                df_with_gaps = self.detect_price_gaps(df_with_indicators, timeframe)
                if df_with_gaps is None:
                    logger.error(f"{timeframe} için fiyat boşlukları tespit edilemedi")
                    continue
                
                # Seans bilgilerini ekle
                logger.info(f"{timeframe} için seans bilgileri ekleniyor...")
                final_df = self.add_session_info(df_with_gaps)
                if final_df is None:
                    logger.error(f"{timeframe} için seans bilgileri eklenemedi")
                    continue
                
                # Son kontroller
                if final_df.isnull().values.any():
                    logger.warning(f"{timeframe} için son işlemlerden sonra eksik veriler var. Dolduruluyor...")
                    final_df = final_df.fillna(method='ffill').fillna(method='bfill')
                
                # Zaman dilimi bilgisini ekle
                final_df['timeframe'] = timeframe
                processed_data[timeframe] = final_df
                
                logger.info(f"{timeframe} verisi başarıyla işlendi. Satır sayısı: {len(final_df)}")
            
            if not processed_data:
                raise Exception("Hiçbir veri işlenemedi")
            
            # Tüm verileri birleştir
            combined_data = pd.concat(processed_data.values(), ignore_index=True)
            
            # Zaman sütununu düzenle
            combined_data['time'] = pd.to_datetime(combined_data['time'])
            combined_data = combined_data.sort_values('time')
            
            logger.info(f"Toplam işlenmiş veri boyutu: {len(combined_data)} satır")
            return combined_data
            
        except Exception as e:
            logger.error(f"Veri işleme hatası: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def get_training_data(self, timeframe=None):
        """
        Eğitim verilerini hazırlar ve döndürür
        
        Args:
            timeframe (str, optional): İstenen zaman dilimi. None ise tüm zaman dilimleri için veri döndürür.
        
        Returns:
            pd.DataFrame: İşlenmiş eğitim verisi
        """
        try:
            # MT5 bağlantı durumunu kontrol et
            if not self.mt5_connector.connected:
                logger.error("MT5 bağlantısı yok. Bağlantı yeniden deneniyor...")
                if not self.mt5_connector.connect():
                    logger.error("MT5 bağlantısı kurulamadı!")
                    return None

            # Tüm timeframe'ler için veriyi bir kerede al
            all_data = {}
            for tf, periods in self.lookback_periods.items():
                if timeframe and tf != timeframe:
                    continue
                
                logger.info(f"{tf} için {periods} mum verisi alınıyor...")
                
                # Veri alımını 3 kez dene
                for attempt in range(3):
                    try:
                        data = self.mt5_connector.get_historical_data(
                            symbol="XAUUSD",
                            timeframe=tf,
                            num_candles=periods
                        )
                        
                        if data is not None and len(data) >= periods * 0.5:  # En az %50 veri gerekli
                            # Veri işleme
                            processed_data = self.process_data(data.copy(), tf)
                            if processed_data is None:
                                logger.error(f"{tf} için veri işleme başarısız")
                                continue
                                
                            all_data[tf] = processed_data
                            logger.info(f"{tf} için {len(processed_data)} satır veri işlendi")
                            break
                        else:
                            logger.warning(f"Deneme {attempt + 1}: {tf} için yeterli veri alınamadı")
                            if attempt < 2:  # Son deneme değilse bekle
                                time.sleep(2)  # 2 saniye bekle
                    except Exception as e:
                        logger.error(f"Deneme {attempt + 1}: {tf} verisi alınırken hata: {str(e)}")
                        if attempt < 2:
                            time.sleep(2)
                
                if tf not in all_data:
                    logger.error(f"{tf} için veri alınamadı")
                    if timeframe:  # Belirli bir timeframe isteniyorsa ve alınamadıysa hata ver
                        return None
                    
            if not all_data:
                logger.error("Hiçbir zaman dilimi için veri alınamadı")
                return None
            
            # Verileri birleştir
            if timeframe:
                return all_data[timeframe]
            else:
                # Tüm timeframe'leri birleştir
                combined_data = pd.concat(all_data.values(), axis=0)
                if len(combined_data) == 0:
                    logger.error("Birleştirilmiş veri boş")
                    return None
                return combined_data
            
        except Exception as e:
            logger.error(f"Eğitim verisi hazırlanırken hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def prepare_sequences(self, df, sequence_length=60, target_column='close', prediction_steps=1, timeframe='5m'):
        """
        Veri dizilerini hazırla ve torch.Tensor'e dönüştür
        """
        try:
            if df is None or len(df) < sequence_length + prediction_steps:
                self.logger.error(f"Yetersiz veri: {len(df) if df is not None else 0} satır < {sequence_length + prediction_steps}")
                return None, None

            # Özellik sütunlarını kontrol et
            feature_columns = [col for col in self.all_features if col in df.columns]
            if not feature_columns:
                self.logger.error("Özellik sütunları bulunamadı")
                return None, None

            # Özellikleri normalize et
            df_normalized = df[feature_columns].copy()
            if not self.feature_scaler_fitted:
                self.feature_scaler.fit(df_normalized)
                self.feature_scaler_fitted = True
            df_normalized = pd.DataFrame(
                self.feature_scaler.transform(df_normalized),
                columns=feature_columns,
                index=df.index
            )

            sequences = []
            targets = []

            for i in range(len(df_normalized) - sequence_length - prediction_steps + 1):
                # Girdi dizisi
                sequence = df_normalized.iloc[i:(i + sequence_length)][feature_columns].values
                sequences.append(sequence)

                # Hedef değer
                target = df[target_column].iloc[i + sequence_length + prediction_steps - 1]
                targets.append(target)

            try:
                # NumPy dizilerine dönüştür ve veri tipini kontrol et
                sequences = np.array(sequences, dtype=np.float32)
                targets = np.array(targets, dtype=np.float32)
                
                # Normalize öncesi hedef değerlerin aralığını logla
                self.logger.info(f"Normalize öncesi hedef değer aralığı: [{np.min(targets):.6f}, {np.max(targets):.6f}]")
                
                # Hedef değerleri fiyat değişim oranlarına dönüştür (yüzde değişim)
                if len(targets) > 1:
                    # Fiyat değişim oranını hesapla (t ve t-1 arasındaki yüzde değişim)
                    targets_returns = np.diff(targets) / targets[:-1]
                    # Son değerin işlenmesi için bir değer ekle (son değer için 0 kabul edelim)
                    targets_returns = np.append(targets_returns, 0)
                    
                    # Çok büyük değerleri kırp
                    targets_abs_max = np.percentile(np.abs(targets_returns), 99)
                    targets_returns = np.clip(targets_returns, -targets_abs_max, targets_abs_max)
                    
                    # -0.1 ile 0.1 arasına normalize et
                    targets = np.tanh(targets_returns / (targets_abs_max * 0.5)) * 0.1
                else:
                    # Tek bir değer varsa, değişim hesaplanamaz, 0 olarak ayarla
                    targets = np.zeros_like(targets)
                
                self.logger.info(f"Normalize sonrası hedef değer aralığı: [{np.min(targets):.6f}, {np.max(targets):.6f}]")

                # Eksik veya sonsuz değerleri kontrol et
                if np.isnan(sequences).any() or np.isinf(sequences).any():
                    self.logger.warning("Sequences'da NaN veya Inf değerler bulundu, temizleniyor...")
                    sequences = np.nan_to_num(sequences, nan=0.0, posinf=1.0, neginf=-1.0)

                if np.isnan(targets).any() or np.isinf(targets).any():
                    self.logger.warning("Targets'da NaN veya Inf değerler bulundu, temizleniyor...")
                    targets = np.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=-1.0)

                # PyTorch tensörlerine dönüştür
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                sequences_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
                targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

                # Tensor boyutlarını ve değer aralıklarını kontrol et
                self.logger.info(f"Sequences tensor shape: {sequences_tensor.shape}")
                self.logger.info(f"Targets tensor shape: {targets_tensor.shape}")
                self.logger.info(f"Sequences değer aralığı: [{sequences_tensor.min().item():.4f}, {sequences_tensor.max().item():.4f}]")
                self.logger.info(f"Targets değer aralığı: [{targets_tensor.min().item():.4f}, {targets_tensor.max().item():.4f}]")

                return sequences_tensor, targets_tensor

            except Exception as e:
                self.logger.error(f"Tensor dönüşümü sırasında hata: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return None, None

        except Exception as e:
            self.logger.error(f"Veri dizileri hazırlanırken hata: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None

    def prepare_rl_state(self, df):
        """
        RL modeli için durum hazırlama
        
        Args:
            df (DataFrame): İşlenmiş veri çerçevesi
            
        Returns:
            numpy.ndarray: RL state
        """
        # Son 60 satırı al
        lookback = 60
        if len(df) < lookback:
            self.logger.error(f"Yetersiz veri: {len(df)} < {lookback}")
            return None
        
        df_tail = df.tail(lookback).copy()
        
        # Gerekli özellikleri seç (RL için sadece fiyat ve teknik indikatörler)
        features = self.price_features + self.technical_indicators
        
        # Eğer bu özellikler yoksa hesapla
        if not all(f in df_tail.columns for f in features):
            df_tail = self._calculate_indicators(df_tail)
        
        # Özellikleri normalize et
        for col in features:
            if col in df_tail.columns:
                df_tail[col] = (df_tail[col] - df_tail[col].min()) / (df_tail[col].max() - df_tail[col].min() + 1e-8)
        
        # RL state'ini oluştur
        state = df_tail[features].values
        
        return state
    
    def get_latest_data(self, timeframe='5m', num_candles=500):
        """
        En güncel veriyi al ve işle
        
        Args:
            timeframe (str): Zaman dilimi ('5m', '15m', '1h')
            num_candles (int): Alınacak mum sayısı
            
        Returns:
            DataFrame: İşlenmiş veri çerçevesi
        """
        try:
            # MT5 connector'ı kontrol et
            if not self.mt5_connector or not hasattr(self.mt5_connector, 'connected') or not self.mt5_connector.connected:
                self.logger.error("MT5 bağlantısı yok")
                return None
                
            # Veriyi al
            data = self.mt5_connector.get_historical_data("XAUUSD", timeframe, num_candles=num_candles)
            if data is None or len(data) < 30:  # Minimum veri kontrolü
                self.logger.error(f"Yeterli veri alınamadı")
                return None
                
            # Veriyi işle
            processed_data = self.process_data(data, timeframe)
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Veri alınırken hata: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None