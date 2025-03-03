import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import logging

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
        
    def add_technical_indicators(self, df):
        """Adds technical indicators"""
        try:
            # Check cache first using DataFrame start/end time as key
            if len(df) > 0:
                cache_key = f"{df['time'].min()}_{df['time'].max()}_{len(df)}"
                if cache_key in self.indicators_cache:
                    logger.debug(f"Using cached indicators for {cache_key}")
                    return self.indicators_cache[cache_key]
            
            # Make a copy to avoid modifying the original DataFrame
            df = df.copy()
            
            # Minimum data check
            min_periods = max(26, 20, 14)  # Maximum of MACD(26), BB(20), and ATR(14) periods
            if len(df) < min_periods:
                logger.warning(f"Not enough data for technical indicators (need at least {min_periods} periods)")
                print(f"Warning: Not enough data for technical indicators (need at least {min_periods} periods)")
                return df
            
            # Fill any existing NaN values before calculations
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Use optimized calculation functions when data size is large
            if len(df) > 1000:
                logger.debug(f"Using optimized calculations for {len(df)} rows")
                return self._add_technical_indicators_optimized(df)
            
            # RSI (14 periods)
            try:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)
                
                # Use minimum periods to handle initial data
                gain_avg = gain.rolling(window=14, min_periods=1).mean()
                loss_avg = loss.rolling(window=14, min_periods=1).mean()
                
                # Handle potential division by zero
                rs = gain_avg / loss_avg.replace(0, 0.001)
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Ensure RSI values are in valid range (0-100)
                df['RSI'] = df['RSI'].clip(0, 100)
                
                # Fill any NaN values
                if df['RSI'].isnull().any():
                    df['RSI'] = df['RSI'].fillna(50)  # Default to neutral RSI
            except Exception as e:
                logger.error(f"Error in RSI calculation: {str(e)}")
                # Create default RSI values
                df['RSI'] = 50  # Default to neutral RSI value
            
            # MACD (12, 26, 9)
            exp1 = df['close'].ewm(span=12, adjust=False, min_periods=5).mean()
            exp2 = df['close'].ewm(span=26, adjust=False, min_periods=5).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False, min_periods=5).mean()
            
            # Bollinger Bands (20 periods)
            df['MA20'] = df['close'].rolling(window=20, min_periods=5).mean()
            df['20dSTD'] = df['close'].rolling(window=20, min_periods=5).std()
            df['Upper_Band'] = df['MA20'] + (df['20dSTD'] * 2)
            df['Lower_Band'] = df['MA20'] - (df['20dSTD'] * 2)
            
            # ATR (14 periods) - Daha güçlü hesaplama
            try:
                # Ensure we have previous day's close
                if len(df) > 1:
                    high_low = df['high'] - df['low']
                    high_close = np.abs(df['high'] - df['close'].shift(1))
                    low_close = np.abs(df['low'] - df['close'].shift(1))
                    
                    # Replace NaNs with 0 in the first row
                    high_close.iloc[0] = high_low.iloc[0]
                    low_close.iloc[0] = high_low.iloc[0]
                    
                    # Calculate True Range
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    ranges = ranges.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    true_range = np.max(ranges, axis=1)
                    
                    # Calculate ATR with more robust method
                    df['ATR'] = true_range.rolling(window=14, min_periods=5).mean()
                    
                    # Alternative ATR calculation if we still have NaNs
                    if df['ATR'].isnull().any():
                        logger.info("Using alternative ATR calculation")
                        # Use simple average of true range for first 14 periods
                        df['ATR'] = true_range.ewm(span=14, adjust=False, min_periods=5).mean()
                else:
                    # Not enough data for ATR
                    logger.warning("Not enough data for ATR calculation (need at least 2 rows)")
                    # Use range of the first candle as a fallback
                    df['ATR'] = df['high'] - df['low']
            except Exception as e:
                logger.error(f"Error in ATR calculation: {str(e)}")
                # Create a simpler ATR as fallback - use 1% of closing price
                df['ATR'] = df['close'] * 0.01
            
            # Check for NaN values in ATR and fix them
            if df['ATR'].isnull().any():
                logger.warning(f"NaN values in ATR calculation. NaN count: {df['ATR'].isnull().sum()}")
                # Calculate the mean of non-NaN ATR values
                mean_atr = df['ATR'].mean()
                if np.isnan(mean_atr):
                    # If all ATRs are NaN, use 1% of price
                    mean_atr = df['close'].mean() * 0.01
                # Replace NaN values with the mean ATR
                df['ATR'] = df['ATR'].fillna(mean_atr)
            
            # Final check and fill for all indicators
            for col in ['RSI', 'MACD', 'Signal_Line', 'MA20', 'Upper_Band', 'Lower_Band', 'ATR']:
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Cache the result if it's valid
            if len(df) > 0 and 'time' in df.columns:
                cache_key = f"{df['time'].min()}_{df['time'].max()}_{len(df)}"
                self.indicators_cache[cache_key] = df.copy()
                
                # Manage cache size
                if len(self.indicators_cache) > self.cache_max_size:
                    # Remove oldest item
                    oldest_key = next(iter(self.indicators_cache))
                    del self.indicators_cache[oldest_key]
                    
            return df
            
        except Exception as e:
            logger.error(f"Error in add_technical_indicators: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Create default values for technical indicators
            for indicator in ['RSI', 'MACD', 'Signal_Line', 'MA20', 'Upper_Band', 'Lower_Band', 'ATR']:
                if indicator not in df.columns:
                    if indicator == 'ATR':
                        df[indicator] = df['close'] * 0.01  # 1% of price as ATR
                    else:
                        df[indicator] = 0
            
            return df
            
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
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Check if enough data
            if len(df) < sequence_length + 20:  # At least 20 bars needed for technical indicators
                print("Warning: Not enough data for technical indicators")
                return torch.FloatTensor([]), torch.FloatTensor([])
            
            # Fill NaN values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
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