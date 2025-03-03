import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import logging
import ta

logger = logging.getLogger("TradingBot.DataProcessor")

class DataProcessor:
    def __init__(self):
        self.price_scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.feature_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        self.all_features = [
            'open', 'high', 'low', 'close', 'tick_volume',  # 5 price features
            'RSI', 'MACD', 'Signal_Line', 'ATR', 'Upper_Band', 'Lower_Band', 'MA20'  # 7 technical indicators
        ]  # Total 12 features + 3 account state = 15 features
        
        # Cache for technical indicators
        self.indicators_cache = {}
        self.cache_max_size = 10  # Maximum number of DataFrames to keep in cache
        self.feature_scaler_fitted = False
        
    def add_technical_indicators(self, df):
        """Add technical indicators to DataFrame"""
        try:
            # Gerekli sütunların varlığını kontrol et
            required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Kritik sütunlar eksik: {missing_columns}")
                return None
            
            df = df.copy()
            
            # Minimum veri kontrolü
            if len(df) < 30:  # En az 30 mum gerekli
                logger.error("Teknik göstergeler için yeterli veri yok")
                return None
            
            # RSI hesaplama
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD hesaplama
            macd = ta.trend.macd(df['close'])
            df['MACD'] = macd
            df['Signal_Line'] = ta.trend.macd_signal(df['close'])
            
            # ATR hesaplama
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Bollinger Bands
            df['MA20'] = ta.trend.sma_indicator(df['close'], window=20)
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['Upper_Band'] = bollinger.bollinger_hband()
            df['Lower_Band'] = bollinger.bollinger_lband()
            
            # NaN değerleri kontrol et
            if df[self.all_features].isnull().any().any():
                logger.error("Teknik göstergelerde NaN değerler var")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Teknik göstergeler hesaplanırken hata: {str(e)}")
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
    
    def prepare_rl_state(self, current_data):
        """
        Prepares state for RL model.
        """
        try:
            # Convert to DataFrame if needed
            if not isinstance(current_data, pd.DataFrame):
                if isinstance(current_data, pd.Series):
                    df = pd.DataFrame([current_data])
                else:
                    print("Warning: current_data is not a DataFrame or Series")
                    # Create a dummy DataFrame with all features
                    dummy_data = {feature: [0] for feature in self.feature_columns}
                    df = pd.DataFrame(dummy_data)
            else:
                df = current_data.copy()
            
            # Minimum data check - for standalone current data only
            if isinstance(df, pd.DataFrame) and len(df) == 1:
                # We just need to ensure it has all columns
                for feature in self.feature_columns:
                    if feature not in df.columns:
                        print(f"Warning: Missing feature: {feature}")
                        df[feature] = 0
                
                # Make sure all necessary indicators are present
                required_indicators = ['RSI', 'MACD', 'Signal_Line', 'ATR', 'Upper_Band', 'Lower_Band', 'MA20']
                for indicator in required_indicators:
                    if indicator not in df.columns:
                        print(f"Warning: Missing indicator: {indicator}")
                        # Set default values for missing indicators
                        if indicator == 'RSI':
                            df[indicator] = 50  # Neutral RSI
                        elif indicator == 'ATR':
                            df[indicator] = df['close'].values[0] * 0.01  # 1% of price
                        else:
                            df[indicator] = 0
            
            # Check and fill NaN values
            if df.isnull().values.any():
                print("Warning: NaN values found, filling with forward/backward fill")
                df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Double check all required features exist
            missing_features = set(self.all_features) - set(df.columns)
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                for feature in missing_features:
                    if feature == 'ATR':
                        df[feature] = df['close'].values[0] * 0.01  # 1% of price as ATR
                    else:
                        df[feature] = 0
            
            # Get and normalize features
            state = df[self.all_features].values
            
            # Initialize scaler if needed
            if not hasattr(self.feature_scaler, 'n_features_in_'):
                # Create a sample of the expected shape for fitting
                sample_data = np.zeros((10, len(self.all_features)))
                sample_data[:, 0:5] = np.random.rand(10, 5) * 2000  # random price data
                sample_data[:, 5:] = np.random.rand(10, 7)  # random indicator data
                self.feature_scaler.fit(sample_data)
            
            # Check for infinities and NaNs
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Scale the data
            try:
                scaled_state = self.feature_scaler.transform(state)
            except Exception as e:
                print(f"Error scaling state: {str(e)}")
                # Create a fallback state
                scaled_state = np.zeros((1, len(self.all_features)))
            
            # Add account state (normalized)
            account_state = np.array([[1.0, 0.0, 0.0]])  # [account_active, position_active, position_type]
            
            # Combine all states
            full_state = np.concatenate([scaled_state.flatten(), account_state.flatten()])
            
            # Final checks
            if len(full_state) != 15:
                print(f"Warning: State vector has {len(full_state)} features, expected 15")
                # Ensure we have exactly 15 features
                if len(full_state) < 15:
                    full_state = np.pad(full_state, (0, 15 - len(full_state)), 'constant')
                else:
                    full_state = full_state[:15]
            
            # Final NaN check
            if np.isnan(full_state).any():
                print("Warning: NaN values in state vector, replacing with zeros")
                full_state = np.nan_to_num(full_state, nan=0.0)
            
            return torch.FloatTensor(full_state)
            
        except Exception as e:
            print(f"Error in prepare_rl_state: {str(e)}")
            import traceback
            traceback.print_exc()
            return torch.zeros(15)  # Return zero state in case of error
    
    def inverse_transform_price(self, scaled_price):
        """Converts normalized price back to real value"""
        return self.price_scaler.inverse_transform([[0, 0, 0, scaled_price, 0]])[0][3]
    
    def prepare_prediction_data(self, df):
        """Prepares data for prediction"""
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
                logger.error("Bazı özellikler eksik")
                return None
            
            # Son veriyi al ve ölçeklendir
            latest_data = df.iloc[-1:][self.all_features]
            
            # Feature scaler'ı güncelle ve veriyi ölçeklendir
            if not self.feature_scaler_fitted:
                self.feature_scaler.fit(df[self.all_features])
                self.feature_scaler_fitted = True
            
            scaled_data = self.feature_scaler.transform(latest_data)
            
            # PyTorch tensor'a çevir
            tensor_data = torch.FloatTensor(scaled_data).unsqueeze(0)
            
            return tensor_data
            
        except Exception as e:
            logger.error(f"Tahmin verisi hazırlanırken hata: {str(e)}")
            return None 