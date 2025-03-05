import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import logging
import ta
import pytz
from datetime import datetime, time

logger = logging.getLogger("TradingBot.DataProcessor")

# Seans zaman dilimlerini tanımla (UTC)
ASIA_SESSION = (time(0, 0), time(9, 0))
EUROPE_SESSION = (time(9, 0), time(17, 0))
US_SESSION = (time(17, 0), time(23, 59))

class DataProcessor:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.feature_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        self.all_features = [
            'open', 'high', 'low', 'close', 'tick_volume',  # 5 price features
            'RSI', 'MACD', 'Signal_Line', 'ATR', 'Upper_Band', 'Lower_Band', 'MA20',  # 7 technical indicators
            'gap', 'gap_size', 'session_asia', 'session_europe', 'session_us'  # 5 yeni özellik
        ]  # Total 17 features + 3 account state = 20 features
        
        # Cache for technical indicators
        self.indicators_cache = {}
        self.cache_max_size = 10  # Maximum number of DataFrames to keep in cache
        self.feature_scaler_fitted = False
        
    def detect_price_gaps(self, df):
        """
        Fiyat boşluklarını (gaps) tespit eder ve ilgili özellikleri ekler
        
        Parametreler:
        - df: İşlenecek DataFrame
        
        Dönüş:
        - Fiyat boşluğu özellikleri eklenmiş DataFrame
        """
        try:
            if df is None:
                logger.warning("detect_price_gaps'a None değeri gönderildi")
                dummy_df = pd.DataFrame()
                dummy_df['gap'] = []
                dummy_df['gap_size'] = []
                return dummy_df
            
            if 'time' not in df.columns:
                logger.debug("Zaman sütunu bulunamadı, fiyat boşlukları tespit edilemedi")
                df['gap'] = 0
                df['gap_size'] = 0
                return df
            
            # Veriyi zaman sırasına göre sırala
            df = df.sort_values('time').copy()
            
            # ATR kolonu yoksa hesapla
            if 'ATR' not in df.columns:
                df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
                # NaN değerleri ffill/bfill ile doldur
                df['ATR'] = df['ATR'].fillna(method='ffill').fillna(method='bfill')
            
            # Sıfır veya NaN ATR değerlerini kontrol et
            min_non_zero_atr = df[df['ATR'] > 0]['ATR'].min() if any(df['ATR'] > 0) else 0.0001
            df['ATR'] = df['ATR'].replace([0, np.nan], min_non_zero_atr)
            
            # ATR <= 0 kontrolü (sonsuz değerleri önlemek için kritik)
            if (df['ATR'] <= 0).any():
                logger.warning(f"ATR sıfır veya negatif değerler içeriyor. Toplam: {(df['ATR'] <= 0).sum()} adet")
                df['ATR'] = df['ATR'].replace([0, np.nan, -np.inf, np.inf], min_non_zero_atr)
            
            # Zaman farkını hesapla
            df['time_diff'] = df['time'].diff().dt.total_seconds() / 60  # Dakika cinsinden
            
            # Normalde zaman farkı sıfırdan büyük olmalı. İlk satırda zaman farkı NaN olacak
            df['time_diff'] = df['time_diff'].fillna(0)
            
            # Normal zaman farkı modunu bul (genelde 1, 5, 15, 30, 60, 240 dakika veya 1440 dakika)
            mode_time_diff = df['time_diff'].value_counts().idxmax()
            logger.debug(f"Normal zaman farkı: {mode_time_diff} dakika")
            
            # Fiyat farkını ATR'ye göre normalize et
            df['norm_price_diff'] = abs(df['open'] - df['close'].shift(1)) / df['ATR']
            
            # Sonsuz ve NaN değerleri makul bir maksimum değere değiştir
            max_allowed_gap = 10.0  # Makul bir maksimum değer
            df['norm_price_diff'] = df['norm_price_diff'].replace([np.inf, -np.inf], max_allowed_gap)
            df['norm_price_diff'] = df['norm_price_diff'].fillna(0)  # İlk satırda fiyat farkı NaN olacak
            
            # Gap olup olmadığını belirle
            # Burada bir gap olması için iki kriter var:
            # 1. Zaman farkı normal farktan büyük olmalı (mesela 4 saatlik timeframe'de 4 saatten fazla)
            # 2. Normalize edilmiş fiyat farkı anlamlı olmalı (ATR'nin en az 0.5 katı)
            # Bu kriterler değişebilir
            
            # Gap kriteri: Zaman farkı normal zaman farkının 1.5 katından büyük VE normalize fiyat farkı 0.5'ten büyük
            df['gap'] = ((df['time_diff'] > mode_time_diff * 1.5) & (df['norm_price_diff'] > 0.5)).astype(int)
            
            # Gap boyutunu hesapla, gap olmayan satırlar için 0
            df['gap_size'] = df['gap'] * df['norm_price_diff']
            
            # Önemli gap'leri loglama
            if df['gap'].sum() > 0:
                significant_gaps = df[df['gap'] == 1]
                for idx, row in significant_gaps.iterrows():
                    gap_time = row['time']
                    gap_size = row['gap_size']
                    if gap_size > 2:  # Büyük gap: ATR'nin 2 katından fazla
                        logger.warning(f"Büyük fiyat gap'i tespit edildi: {gap_time}, Büyüklük: {gap_size:.2f} ATR")
                    elif gap_size > 1:  # Anlamlı gap: ATR'nin 1-2 katı arası
                        logger.info(f"Anlamlı fiyat gap'i tespit edildi: {gap_time}, Büyüklük: {gap_size:.2f} ATR")
                    else:  # Küçük gap: ATR'nin 0.5-1 katı arası
                        logger.debug(f"Küçük fiyat gap'i tespit edildi: {gap_time}, Büyüklük: {gap_size:.2f} ATR")
            
            # Geçici sütunu kaldır
            if 'time_diff' in df.columns:
                df = df.drop(columns=['time_diff'])
            
            # Son kontrol: NaN veya sonsuz değerlerin kontrolünü yap
            if df['gap_size'].isnull().any() or np.isinf(df['gap_size']).any():
                logger.warning("gap_size sütununda NaN veya sonsuz değerler var. Temizleniyor...")
                df['gap_size'] = df['gap_size'].replace([np.inf, -np.inf], max_allowed_gap)
                df['gap_size'] = df['gap_size'].fillna(0)
            
            # Gap size'ı maksimum değere kırp
            if (df['gap_size'] > max_allowed_gap).any():
                logger.warning(f"Bazı gap_size değerleri maksimum değer {max_allowed_gap}'dan büyük. Kırpılıyor...")
                df['gap_size'] = df['gap_size'].clip(0, max_allowed_gap)
            
            return df
        
        except Exception as e:
            logger.error(f"Fiyat boşlukları tespit edilirken hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Hata durumunda varsayılan değerlerle devam et
            # Bu kısımda df None olabilir, kontrol et
            if df is not None:
                df['gap'] = 0
                df['gap_size'] = 0
                return df
            else:
                # DataFrame none ise boş bir dataframe oluştur
                dummy_df = pd.DataFrame()
                dummy_df['gap'] = []
                dummy_df['gap_size'] = []
                return dummy_df
    
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
                logger.debug("Zaman sütunu bulunamadı, seans bilgileri eklenemedi")
                df['session_asia'] = 0
                df['session_europe'] = 0
                df['session_us'] = 0
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
            df['session_asia'] = df.apply(lambda x: 1 if ASIA_SESSION[0] <= time(x['hour'], x['minute']) < ASIA_SESSION[1] else 0, axis=1)
            df['session_europe'] = df.apply(lambda x: 1 if EUROPE_SESSION[0] <= time(x['hour'], x['minute']) < EUROPE_SESSION[1] else 0, axis=1)
            df['session_us'] = df.apply(lambda x: 1 if US_SESSION[0] <= time(x['hour'], x['minute']) <= US_SESSION[1] else 0, axis=1)
            
            # Seans dağılımını logla (sadece debug seviyesinde)
            logger.debug(f"Seans dağılımı: Asya: {df['session_asia'].mean()*100:.1f}%, "
                       f"Avrupa: {df['session_europe'].mean()*100:.1f}%, "
                       f"ABD: {df['session_us'].mean()*100:.1f}%")
            
            # Geçici sütunları kaldır
            df = df.drop(['time_utc', 'hour', 'minute'], axis=1, errors='ignore')
            
            return df
            
        except Exception as e:
            logger.error(f"Seans bilgileri eklenirken hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Hata durumunda varsayılan değerlerle devam et
            df['session_asia'] = 0
            df['session_europe'] = 0
            df['session_us'] = 0
            return df
    
    def add_technical_indicators(self, df):
        """
        Mum verilerine teknik göstergeleri ekler
        
        Parametreler:
        - df: İşlenecek DataFrame
        
        Dönüş:
        - Teknik göstergeler eklenmiş DataFrame
        """
        try:
            if df is None:
                logger.warning("add_technical_indicators'a None değeri gönderildi")
                return None
            
            # Gerekli sütunların varlığını kontrol et
            required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Eksik sütunlar tespit edildi: {missing_columns}. Otomatik olarak ekleniyor.")
                df = df.copy()
                
                # Eksik sütunları ekle
                for col in missing_columns:
                    if col == 'tick_volume':
                        # tick_volume yoksa, varsayılan değer olarak 1 ata
                        df['tick_volume'] = 1
                    elif col in ['open', 'high', 'low'] and 'close' in df.columns:
                        # Eğer close varsa, diğer fiyat sütunları için close değerini kullan
                        df[col] = df['close']
                    elif col == 'close' and 'open' in df.columns:
                        # Eğer open varsa, close için open değerini kullan
                        df[col] = df['open']
                    else:
                        # Diğer durumlar için varsayılan değer
                        logger.error(f"Kritik sütun {col} için varsayılan değer atanamıyor")
                        return None
            else:
                df = df.copy()
            
            # Minimum veri kontrolü
            if len(df) < 30:  # En az 30 mum gerekli
                logger.error("Teknik göstergeler için yeterli veri yok")
                return None
            
            # Piyasa boşluklarını (gap) kontrol et
            if 'time' in df.columns:
                df = df.sort_values('time')
                # Zaman farkını hesapla
                df['time_diff'] = df['time'].diff().dt.total_seconds() / 60  # Dakika cinsinden
                
                # Standart zaman aralığını bul (en yaygın zaman farkı)
                if len(df) > 1:
                    # En yaygın zaman farkını bul (mod)
                    time_diff_mode = df['time_diff'].value_counts().idxmax()
                    
                    # Büyük boşlukları tespit et (standart aralığın 3 katından fazla)
                    gaps = df[df['time_diff'] > time_diff_mode * 3]
                    
                    if not gaps.empty:
                        logger.warning(f"Veri setinde {len(gaps)} adet piyasa boşluğu (gap) tespit edildi")
                        logger.warning(f"En büyük boşluk: {gaps['time_diff'].max()} dakika")
                        logger.warning(f"Boşlukların olduğu tarihler: {gaps['time'].tolist()}")
                
                # Geçici sütunu kaldır
                if 'time_diff' in df.columns:
                    df = df.drop('time_diff', axis=1)
            
            # Verilerdeki NaN veya sonsuz değerleri tespit et ve düzelt
            # Temiz veri elde ettiğimizden emin olalım
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    # Sonsuz değerleri son geçerli değer ile değiştir
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # NaN değerleri interpolasyon ile doldur
                    df[col] = df[col].interpolate(method='linear').ffill().bfill()
            
            # RSI hesaplama
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD hesaplama
            macd = ta.trend.macd(df['close'])
            df['MACD'] = macd
            df['Signal_Line'] = ta.trend.macd_signal(df['close'])
            
            # ATR hesaplama
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # ATR değerinin sıfır olduğu durumları kontrol et
            min_non_zero_atr = df[df['ATR'] > 0]['ATR'].min() if any(df['ATR'] > 0) else 0.0001
            df['ATR'] = df['ATR'].replace([0, np.nan, np.inf, -np.inf], min_non_zero_atr)
            
            # Bollinger Bands
            df['MA20'] = ta.trend.sma_indicator(df['close'], window=20)
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['Upper_Band'] = bollinger.bollinger_hband()
            df['Lower_Band'] = bollinger.bollinger_lband()
            
            # İlk hesaplama sonrasında göstergelerdeki NaN değerleri temizle
            # RSI için özel işlem - ilk değerler 50 (nötr) olsun
            if 'RSI' in df.columns and df['RSI'].isnull().any():
                # Önce backward fill (sonraki değerlerden ilk değerleri doldur)
                df['RSI'] = df['RSI'].fillna(method='bfill')
                # Kalan NaN'ları 50 ile doldur
                df['RSI'] = df['RSI'].fillna(50)
            
            # MACD ve Signal Line için özel işlem
            for col in ['MACD', 'Signal_Line']:
                if col in df.columns and df[col].isnull().any():
                    # Backward fill ile NaN'ları temizlemeye çalış
                    df[col] = df[col].fillna(method='bfill')
                    # Kalan NaN'ları 0 ile doldur (nötr değer)
                    df[col] = df[col].fillna(0)
            
            # Bollinger Bands için özel işlem
            for col in ['Upper_Band', 'Lower_Band', 'MA20']:
                if col in df.columns and df[col].isnull().any():
                    # Önce backward fill ile doldur
                    df[col] = df[col].fillna(method='bfill')
                    # Kalan NaN'lar için kapanış fiyatına dayalı değerler kullan
                    if col == 'Upper_Band':
                        df[col] = df[col].fillna(df['close'] * 1.02)  # Üst bant kapanışın %2 üstü
                    elif col == 'Lower_Band':
                        df[col] = df[col].fillna(df['close'] * 0.98)  # Alt bant kapanışın %2 altı
                    elif col == 'MA20':
                        df[col] = df[col].fillna(df['close'])  # MA basitçe kapanış fiyatı olsun
            
            # Fiyat boşluklarını (gaps) tespit et ve ekle
            df = self.detect_price_gaps(df)
            
            # Seans bilgilerini ekle
            df = self.add_session_info(df)
            
            # NaN değerleri tespit et ve raporla
            nan_cols = [col for col in self.all_features if col in df.columns and df[col].isnull().any()]
            if nan_cols:
                nan_counts = {col: df[col].isnull().sum() for col in nan_cols}
                logger.warning(f"Teknik göstergelerde NaN değerler var: {nan_cols}")
                logger.warning(f"NaN değer sayıları: {nan_counts}")
                
                # NaN değerleri temizle - önce forward fill, sonra backward fill, yine de NaN kalırsa makul değerlerle doldur
                # Her sütunu ayrı ayrı temizle
                for col in nan_cols:
                    # Önce ileriye doğru doldur
                    df[col] = df[col].fillna(method='ffill')
                    # Sonra geriye doğru doldur
                    df[col] = df[col].fillna(method='bfill')
                
                # Yine de NaN değerler varsa, makul varsayılanlarla doldur
                default_values = {
                    'RSI': 50,        # Nötr
                    'MACD': 0,        # Nötr
                    'Signal_Line': 0, # Nötr
                    'Upper_Band': df['close'].mean() * 1.02 if 'close' in df.columns else 1,  # Ortalama +%2
                    'Lower_Band': df['close'].mean() * 0.98 if 'close' in df.columns else 1,  # Ortalama -%2
                    'MA20': df['close'].mean() if 'close' in df.columns else 1,  # Ortalama
                    'gap_size': 0,    # Gap yok
                    'ATR': min_non_zero_atr,  # Minimum ATR
                    'norm_price_diff': 0  # Nötr
                }
                
                for col in nan_cols:
                    if col in default_values:
                        df[col] = df[col].fillna(default_values[col])
                    else:
                        # Diğer sütunlar için varsayılan 0
                        df[col] = df[col].fillna(0)
                
                # Son bir kontrol - hala NaN var mı?
                remaining_nans = df[self.all_features].isnull().sum().sum()
                if remaining_nans > 0:
                    logger.warning(f"Tüm temizleme işlemlerine rağmen {remaining_nans} NaN değer kaldı")
                    # Kalan tüm NaN değerleri 0 ile doldur
                    for col in self.all_features:
                        if col in df.columns and df[col].isnull().any():
                            df[col] = df[col].fillna(0)
            
            # Sonsuz değerleri kontrol et
            try:
                # Önce tüm gerekli sütunların olduğundan emin ol
                df = self.all_features_exist(df)
                
                if df is None:
                    logger.error("Veriler kontrol edilirken beklenmeyen bir hata oluştu, None değeri döndürüldü")
                    return None
                
                # all_features listesinde olup df'te olmayan sütunları sıfır ile doldur
                for col in self.all_features:
                    if col not in df.columns:
                        df[col] = 0
                        logger.warning(f"Eksik sütun {col} sıfır ile dolduruldu")
                
                # İnfinite değerleri kontrol et
                if (df[self.all_features].replace([np.inf, -np.inf], np.nan).isnull().any().any()):
                    logger.warning("Teknik göstergelerde sonsuz değerler tespit edildi, temizleniyor")
                    # Sonsuz değerleri, sıfır olmayan en küçük değerle değiştir
                    for col in self.all_features:
                        if col in df.columns:
                            # Makul bir maksimum değer belirle (her sütun için değişebilir)
                            max_val = 10.0 if col in ['gap_size', 'norm_price_diff'] else 100.0
                            
                            # Sonsuz değerleri makul değerlerle değiştir
                            df[col] = df[col].replace([np.inf, -np.inf], max_val)
                            
                            # Değerleri makul bir aralığa sınırla
                            df[col] = df[col].clip(lower=-max_val, upper=max_val)
            except Exception as e:
                logger.warning(f"Sonsuz değerler kontrol edilirken hata: {str(e)}")
                # İşleme devam etmek için hata yönetimi
                import traceback
                logger.debug(traceback.format_exc())
            
            return df
            
        except Exception as e:
            logger.error(f"Teknik göstergeler hesaplanırken hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
    def _add_technical_indicators_optimized(self, df):
        """Optimized version of indicator calculation for large datasets"""
        try:
            # Use numpy operations where possible for better performance with large DataFrames
            close_prices = df['close'].values
            
            # RSI - vectorized calculation
            try:
                delta = np.diff(close_prices, prepend=close_prices[0])
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                # Using rolling calculation with minimum period of 1
                avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean().values
                avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean().values
                
                # Avoid division by zero
                avg_loss = np.where(avg_loss < 0.001, 0.001, avg_loss)
                rs = np.divide(avg_gain, avg_loss)
                
                # Calculate RSI
                rsi = 100 - (100 / (1 + rs))
                
                # Ensure values are in valid range
                rsi = np.clip(rsi, 0, 100)
                
                # Replace NaN values
                rsi = np.nan_to_num(rsi, nan=50.0)
                
                df['RSI'] = rsi
            except Exception as e:
                logger.error(f"Error in RSI calculation (optimized): {str(e)}")
                # Default RSI value
                df['RSI'] = 50
            
            # MACD
            exp1 = pd.Series(close_prices).ewm(span=12, adjust=False, min_periods=5).mean().values
            exp2 = pd.Series(close_prices).ewm(span=26, adjust=False, min_periods=5).mean().values
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = pd.Series(df['MACD'].values).ewm(span=9, adjust=False, min_periods=5).mean().values
            
            # Moving Averages - vectorized
            df['MA20'] = pd.Series(close_prices).rolling(window=20, min_periods=5).mean().values
            
            # Bollinger Bands
            std20 = pd.Series(close_prices).rolling(window=20, min_periods=5).std().values
            df['20dSTD'] = std20
            df['Upper_Band'] = df['MA20'] + (std20 * 2)
            df['Lower_Band'] = df['MA20'] - (std20 * 2)
            
            # ATR calculation
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # Calculate True Range components
            high_low = high_prices - low_prices
            
            # Shift close prices
            close_shifted = np.roll(close_prices, 1)
            close_shifted[0] = close_prices[0]  # Handle first element
            
            high_close = np.abs(high_prices - close_shifted)
            low_close = np.abs(low_prices - close_shifted)
            
            # Get true range - element-wise maximum
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Calculate ATR
            df['ATR'] = pd.Series(true_range).rolling(window=14, min_periods=5).mean().values
            
            # Fill any NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in optimized indicator calculation: {str(e)}")
            # Fall back to standard method
            return self.add_technical_indicators(df)
        
    def prepare_data(self, df, sequence_length=60):
        """Prepares data for LSTM model"""
        try:
            # Önce sütunların var olup olmadığını kontrol et
            required_columns = self.feature_columns.copy()
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Adding missing columns to DataFrame: {missing_columns}")
                for col in missing_columns:
                    if col == 'tick_volume':
                        # tick_volume eksikse, varsayılan değerlerle doldur
                        df[col] = 1
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Check if enough data
            if len(df) < sequence_length + 20:  # At least 20 bars needed for technical indicators
                print("Warning: Not enough data for technical indicators")
                return torch.FloatTensor([]), torch.FloatTensor([])
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Ensure all required columns exist
            for col in self.feature_columns:
                if col not in df.columns:
                    print(f"Error: Required column '{col}' is missing even after preprocessing")
                    return torch.FloatTensor([]), torch.FloatTensor([])
            
            # Normalize price data
            price_data = df[self.feature_columns].values
            scaled_prices = self.price_scaler.fit_transform(price_data)
            
            # Prepare numpy arrays
            sequences = []
            targets = []
            
            for i in range(len(scaled_prices) - sequence_length):
                sequence = scaled_prices[i:(i + sequence_length)]
                target = scaled_prices[i + sequence_length][3]  # close price
                sequences.append(sequence)
                targets.append(target)
            
            # Create numpy arrays directly instead of lists
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            return torch.FloatTensor(sequences), torch.FloatTensor(targets)
            
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            return torch.FloatTensor([]), torch.FloatTensor([])
    
    def prepare_data_with_weights(self, df, sequence_length=60, train_split=0.8, target_column='close', prediction_steps=1, weight_recent_factor=2.0):
        """
        LSTM modeli için ağırlıklı eğitim verisi hazırlar
        Yeni verilere daha yüksek ağırlık verilir
        
        Parametreler:
        - df: İşlenecek DataFrame
        - sequence_length: Her bir örnek için kullanılacak geçmiş veri miktarı
        - train_split: Eğitim/doğrulama bölme oranı (0.8 = %80 eğitim, %20 doğrulama)
        - target_column: Tahmin edilecek hedef sütun
        - prediction_steps: Kaç adım ilerisini tahmin edeceğiz
        - weight_recent_factor: Yeni verilere verilecek ağırlık çarpanı
        
        Dönüş:
        - X_train, y_train, X_val, y_val, sample_weights
        """
        try:
            # Gerekli sütunların varlığını kontrol et
            required_columns = self.all_features
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Gerekli sütunlar eksik: {missing_columns}")
                return None, None, None, None, None
            
            # Teknik göstergeleri ekle (eğer yoksa)
            if 'RSI' not in df.columns:
                df = self.add_technical_indicators(df)
                if df is None:
                    return None, None, None, None, None
            
            # NaN değerleri doldur
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Minimum veri kontrolü
            if len(df) < sequence_length + prediction_steps:
                logger.error(f"Yeterli veri yok. En az {sequence_length + prediction_steps} satır gerekli, {len(df)} satır mevcut")
                return None, None, None, None, None
            
            # Fiyat verilerini normalize et
            price_data = df[['open', 'high', 'low', 'close']].values
            self.price_scaler.fit(price_data)
            normalized_prices = self.price_scaler.transform(price_data)
            
            # Normalize edilmiş fiyatları DataFrame'e geri koy
            df[['open', 'high', 'low', 'close']] = normalized_prices
            
            # Diğer özellikleri normalize et (tick_volume ve teknik göstergeler)
            feature_data = df[self.all_features].values
            self.feature_scaler.fit(feature_data)
            normalized_features = self.feature_scaler.transform(feature_data)
            
            # Normalize edilmiş özellikleri DataFrame'e geri koy
            df[self.all_features] = normalized_features
            
            # Dizileri hazırla
            X, y = [], []
            
            # Her bir zaman adımı için bir dizi oluştur
            for i in range(len(df) - sequence_length - prediction_steps + 1):
                # Geçmiş veri
                X.append(df[self.all_features].values[i:i+sequence_length])
                # Gelecek fiyat (hedef)
                y.append(df[target_column].values[i+sequence_length+prediction_steps-1])
            
            # NumPy dizilerine dönüştür
            X = np.array(X)
            y = np.array(y)
            
            # Eğitim ve doğrulama setlerine böl
            train_size = int(len(X) * train_split)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Ağırlıkları hesapla (yeni verilere daha yüksek ağırlık ver)
            sample_weights = np.ones(len(X_train))
            
            # Son %20'lik kısma daha yüksek ağırlık ver
            recent_data_start = int(len(X_train) * 0.8)
            
            # Doğrusal artan ağırlıklar (1'den weight_recent_factor'a)
            weights_recent = np.linspace(1, weight_recent_factor, len(X_train) - recent_data_start)
            sample_weights[recent_data_start:] = weights_recent
            
            # Gap ve seans bazlı ağırlıklandırma
            # Gap olan örneklere daha yüksek ağırlık ver
            for i in range(len(X_train)):
                # Son dizideki gap bilgisini kontrol et
                gap_present = X_train[i, -1, self.all_features.index('gap')]
                gap_size = X_train[i, -1, self.all_features.index('gap_size')]
                
                # Gap varsa ağırlığı artır (gap büyüklüğüne göre)
                if gap_present > 0:
                    # Gap büyüklüğüne göre ağırlık artışı (1.5 ile 3 arasında)
                    gap_weight_factor = 1.5 + min(gap_size, 3) / 2
                    sample_weights[i] *= gap_weight_factor
                
                # Seans bazlı ağırlıklandırma
                session_asia = X_train[i, -1, self.all_features.index('session_asia')]
                session_europe = X_train[i, -1, self.all_features.index('session_europe')]
                session_us = X_train[i, -1, self.all_features.index('session_us')]
                
                # Farklı seanslara farklı ağırlıklar ver (örnek: Avrupa seansı daha önemli olabilir)
                if session_asia > 0:
                    sample_weights[i] *= 1.2  # Asya seansı
                if session_europe > 0:
                    sample_weights[i] *= 1.5  # Avrupa seansı (en önemli)
                if session_us > 0:
                    sample_weights[i] *= 1.3  # ABD seansı
            
            # Ağırlıkları normalize et (ortalama 1 olacak şekilde)
            sample_weights = sample_weights * len(sample_weights) / np.sum(sample_weights)
            
            # PyTorch tensor'larına dönüştür
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train).unsqueeze(1)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val).unsqueeze(1)
            sample_weights = torch.FloatTensor(sample_weights)
            
            logger.info(f"Veri hazırlama tamamlandı: {X_train.shape[0]} eğitim örneği, {X_val.shape[0]} doğrulama örneği")
            logger.info(f"Ağırlık aralığı: {sample_weights.min().item():.2f} - {sample_weights.max().item():.2f}")
            
            return X_train, y_train, X_val, y_val, sample_weights
            
        except Exception as e:
            logger.error(f"Veri hazırlanırken hata: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None, None

    def prepare_prediction_data(self, df):
        """
        Tahmin için veri hazırlar
        Fiyat boşluğu ve seans bilgilerini de içerir
        
        Parametreler:
        - df: İşlenecek DataFrame
        
        Dönüş:
        - PyTorch tensor formatında hazırlanmış veri
        """
        try:
            # Veri kontrolü
            if df is None or len(df) == 0:
                logger.error("Veri yok")
                return None
            
            # Teknik göstergeleri ekle
            df = self.add_technical_indicators(df)
            if df is None:
                return None
            
            # Tüm özelliklerin var olduğunu kontrol et
            if not all(feature in df.columns for feature in self.all_features):
                missing_features = [f for f in self.all_features if f not in df.columns]
                logger.error(f"Bazı özellikler eksik: {missing_features}")
                return None
            
            # Son veriyi al ve kontrolden geçir
            latest_data = df.iloc[-1:][self.all_features].copy()
            
            # Sonsuz veya çok büyük değerleri kontrol et ve düzelt
            for col in self.all_features:
                if col in latest_data.columns:
                    # Sonsuz değerleri makul değerlerle değiştir
                    max_val = 10.0 if col in ['gap_size'] else 100.0
                    
                    # Sonsuz değerleri temizle
                    latest_data[col] = latest_data[col].replace([np.inf, -np.inf], np.nan)
                    
                    # NaN değerleri temizle
                    if latest_data[col].isnull().any():
                        # Eğer df'de daha fazla veri varsa, son geçerli değeri kullan
                        if len(df) > 1:
                            last_valid = df[col].dropna().iloc[-1] if not df[col].dropna().empty else 0
                            latest_data[col] = latest_data[col].fillna(last_valid)
                        else:
                            # Varsayılan değerler ata
                            if col == 'RSI':
                                latest_data[col] = latest_data[col].fillna(50)
                            elif col in ['open', 'high', 'low', 'close']:
                                # Fiyat sütunları için mevcut değerlerden doldur
                                if not df['close'].isnull().all():
                                    latest_data[col] = latest_data[col].fillna(df['close'].dropna().iloc[-1])
                                else:
                                    latest_data[col] = latest_data[col].fillna(0)
                            else:
                                latest_data[col] = latest_data[col].fillna(0)
            
            # Değerleri makul bir aralıkta sınırla
            for col in self.all_features:
                if col in latest_data.columns:
                    if col == 'RSI':
                        latest_data[col] = latest_data[col].clip(0, 100)
                    elif col == 'gap_size':
                        latest_data[col] = latest_data[col].clip(0, 10)
                    elif col in ['session_asia', 'session_europe', 'session_us', 'gap']:
                        latest_data[col] = latest_data[col].clip(0, 1)
            
            # Feature scaler'ı güncelle ve veriyi ölçeklendir
            if not self.feature_scaler_fitted:
                try:
                    # Önce sonsuz değerleri ve NaN'ları temizle
                    clean_df = df[self.all_features].replace([np.inf, -np.inf], np.nan)
                    
                    # NaN değerleri doldur
                    clean_df = clean_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
                    # Feature scaler'ı eğit
                    self.feature_scaler.fit(clean_df)
                    self.feature_scaler_fitted = True
                except Exception as e:
                    logger.error(f"Feature scaler eğitilirken hata: {str(e)}")
                    # Basit MinMaxScaler yerine manuel normalizasyon yap
                    for col in self.all_features:
                        if col in latest_data.columns:
                            if col not in ['session_asia', 'session_europe', 'session_us', 'gap']:
                                max_val = df[col].max() if not df[col].isnull().all() else 1
                                min_val = df[col].min() if not df[col].isnull().all() else 0
                                if max_val > min_val:
                                    latest_data[col] = (latest_data[col] - min_val) / (max_val - min_val)
                                else:
                                    latest_data[col] = 0
            
            try:
                # Son kontrol - hala NaN veya sonsuz değer var mı?
                if latest_data.isnull().any().any() or np.isinf(latest_data.values).any():
                    logger.warning("Hala NaN veya sonsuz değerler var, temizleniyor")
                    latest_data = latest_data.replace([np.inf, -np.inf], 0).fillna(0)
                
                # Ölçeklendirme
                scaled_data = self.feature_scaler.transform(latest_data)
                
                # PyTorch tensor'a çevir
                tensor_data = torch.FloatTensor(scaled_data).unsqueeze(0)
                
                # Gap ve seans bilgilerini logla
                gap_present = df.iloc[-1]['gap'] if 'gap' in df.columns else 0
                gap_size = df.iloc[-1]['gap_size'] if 'gap_size' in df.columns else 0
                
                if gap_present > 0:
                    logger.info(f"Tahmin verisi hazırlanırken fiyat boşluğu (gap) tespit edildi. Büyüklük: {gap_size:.2f} ATR")
                
                # Hangi seansta olduğumuzu logla
                if 'session_asia' in df.columns and df.iloc[-1]['session_asia'] > 0:
                    logger.info("Şu anda Asya seansındayız")
                elif 'session_europe' in df.columns and df.iloc[-1]['session_europe'] > 0:
                    logger.info("Şu anda Avrupa seansındayız")
                elif 'session_us' in df.columns and df.iloc[-1]['session_us'] > 0:
                    logger.info("Şu anda ABD seansındayız")
                
                return tensor_data
            except Exception as e:
                logger.error(f"Veri ölçeklendirilirken hata: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Tahmin verisi hazırlanırken hata: {str(e)}")
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
                'RSI': 50,           # Nötr
                'MACD': 0,           # Nötr
                'Signal_Line': 0,    # Nötr
                'ATR': 0.0001,       # Küçük bir değer
                'Upper_Band': df['close'].mean() * 1.02 if 'close' in df.columns else 1,  # Ortalama +%2
                'Lower_Band': df['close'].mean() * 0.98 if 'close' in df.columns else 1,  # Ortalama -%2
                'MA20': df['close'].mean() if 'close' in df.columns else 1,  # Ortalama
                'gap': 0,            # Gap yok
                'gap_size': 0,       # Gap boyutu
                'norm_price_diff': 0, # Normalize fiyat farkı
                'session_asia': 0,   # Seans bilgileri
                'session_europe': 0, # Seans bilgileri
                'session_us': 0      # Seans bilgileri
            }
            
            for col in missing_cols:
                if col in default_values:
                    df[col] = default_values[col]
                else:
                    df[col] = 0  # Diğer bilinmeyen sütunlar için 0
                logger.debug(f"Eksik sütun eklendi: {col}")
        
        return df