import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import MetaTrader5 as mt5

class LSTMPredictor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class ForexTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, lstm_model: LSTMPredictor, initial_balance: float = 10000.0,
                 max_position_size: float = 1.0, transaction_fee: float = 0.0001):
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        self.lstm_model = lstm_model
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_fee = transaction_fee
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),  # price data (5) + technical indicators (7) + account state (3)
            dtype=np.float32
        )
        
        # Action space: 0 (wait), 1 (buy), 2 (sell)
        self.action_space = spaces.Discrete(3)
        
        self.current_step = 0
        self.current_position = 0
        self.current_price = 0
        self.last_trade_price = 0
        
    def _get_observation(self):
        # Get current state
        current_data = self.df.iloc[self.current_step]
        
        # Get price data and technical indicators (5 features)
        price_data = np.array([
            current_data['open'],
            current_data['high'],
            current_data['low'],
            current_data['close'],
            current_data['tick_volume']
        ])
        
        # Technical indicators (7 features)
        technical_indicators = np.array([
            current_data['RSI'],
            current_data['MACD'],
            current_data['Signal_Line'],
            current_data['ATR'],
            current_data['Upper_Band'],
            current_data['Lower_Band'],
            current_data['MA20']
        ])
        
        # Account state (3 features)
        account_state = np.array([
            self.balance / self.initial_balance,  # Normalize balance
            self.current_position,
            self.last_trade_price / current_data['close'] if self.last_trade_price > 0 else 0
        ])
        
        # Safe normalization function
        def safe_normalize(data):
            if len(data) == 0:
                return data
            data_mean = np.mean(data)
            if abs(data_mean) < 1e-8:  # Very close to zero
                return data - np.mean(data)  # Only center
            return (data - data_mean) / (np.std(data) + 1e-8)  # Normalize with standard deviation
        
        # Normalize data
        normalized_price = safe_normalize(price_data)
        normalized_tech = safe_normalize(technical_indicators)
        
        # Combine all data and flatten
        observation = np.concatenate([
            normalized_price,  # 5 features
            normalized_tech,   # 7 features
            account_state     # 3 features
        ]).flatten()  # Total 15 features
        
        # Check for NaN values
        if np.isnan(observation).any():
            observation = np.nan_to_num(observation, nan=0.0)
        
        return observation.astype(np.float32)
    
    def _calculate_reward(self, action: int) -> float:
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0
        
        # Calculate profit/loss if position exists
        if self.current_position != 0:
            # Calculate profit/loss
            price_diff = (current_price - self.current_price) * self.current_position
            transaction_cost = abs(price_diff) * self.transaction_fee
            reward = price_diff - transaction_cost
            
            # Check for loss
            if reward < -self.initial_balance * 0.02:  # More than 2% loss
                reward *= 2  # Increase penalty
            
            # Stop-loss and take-profit checks
            if abs(price_diff / self.current_price) > 0.02:  # 2% change
                if price_diff > 0:
                    reward *= 1.5  # Take-profit bonus
                else:
                    reward *= 2  # Stop-loss penalty
        
        # Position change cost
        if action != 0 and self.current_position != 0:
            reward -= abs(current_price * self.transaction_fee)
        
        # Normalize reward
        reward = np.clip(reward / self.initial_balance, -1, 1)
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Save previous state
        prev_balance = self.balance
        
        # Apply action (-1: sell, 0: wait, 1: buy)
        action = action - 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Update position
        if action != 0:  # If not wait
            # Close current position
            if self.current_position != 0:
                realized_pnl = (self.df.iloc[self.current_step]['close'] - self.current_price) * self.current_position
                self.balance += realized_pnl - abs(realized_pnl) * self.transaction_fee
                self.current_position = 0
                self.current_price = 0
            
            # Open new position
            if action != self.current_position:
                self.current_position = action
                self.current_price = self.df.iloc[self.current_step]['close']
                # Deduct transaction fee
                self.balance -= abs(self.current_price * self.transaction_fee)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.df) - 1) or (self.balance <= self.initial_balance * 0.5)
        
        # Get new observation
        obs = self._get_observation()
        
        # Save info
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.current_position,
            'reward': reward,
            'return': (self.balance - prev_balance) / prev_balance if prev_balance > 0 else 0,
            'total_return': (self.balance - self.initial_balance) / self.initial_balance
        }
        
        return obs, reward, done, False, info
    
    def reset(self, seed=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_position = 0
        self.current_price = 0
        self.current_step = 0
        
        return self._get_observation(), {}

class RLTrader:
    def __init__(self, lstm_model: LSTMPredictor, env_params: dict):
        self.lstm_model = lstm_model
        self.env_params = env_params
        self.model = None
        self.env = None  # Çevre değişkenini burada tanımlıyoruz
        
        # Create initial environment and make it accessible
        self.env = self.create_env(env_params.get('df', pd.DataFrame()))
        
        # Set PPO model parameters
        self.model_params = {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "ent_coef": 0.01
        }
    
    def create_env(self, df):
        """Creates a vectorized environment for RL training"""
        
        # Check if DataFrame is valid
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            print("Warning: Empty DataFrame provided for environment creation. Using dummy data.")
            # Create a minimal dummy DataFrame to initialize environment
            df = pd.DataFrame({
                'open': [100.0] * 10,
                'high': [101.0] * 10,
                'low': [99.0] * 10,
                'close': [100.5] * 10,
                'tick_volume': [100] * 10,
                'RSI': [50.0] * 10,
                'MACD': [0.0] * 10,
                'Signal_Line': [0.0] * 10,
                'ATR': [1.0] * 10,
                'Upper_Band': [102.0] * 10,
                'Lower_Band': [98.0] * 10,
                'MA20': [100.0] * 10
            })
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume', 
                         'RSI', 'MACD', 'Signal_Line', 'ATR', 
                         'Upper_Band', 'Lower_Band', 'MA20']
        
        # Check and add missing columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Adding missing columns to environment DataFrame: {missing_cols}")
            for col in missing_cols:
                if col in ['open', 'high', 'low', 'close']:
                    df[col] = 100.0
                elif col == 'tick_volume':
                    df[col] = 100
                elif col == 'RSI':
                    df[col] = 50.0
                elif col in ['MACD', 'Signal_Line']:
                    df[col] = 0.0
                elif col == 'ATR':
                    df[col] = 1.0
                elif col == 'Upper_Band':
                    df[col] = 102.0
                elif col == 'Lower_Band':
                    df[col] = 98.0
                elif col == 'MA20':
                    df[col] = 100.0
            
        # Create environment creator function   
        def _init():
            try:
                env_params = self.env_params.copy()
                if 'df' in env_params:
                    del env_params['df']
                if 'lstm_model' in env_params:
                    del env_params['lstm_model']
                return ForexTradingEnv(df, self.lstm_model, **env_params)
            except Exception as e:
                print(f"Error creating environment: {str(e)}")
                # Use default parameters if any error occurs
                return ForexTradingEnv(
                    df=df, 
                    lstm_model=self.lstm_model,
                    initial_balance=10000.0,
                    max_position_size=1.0,
                    transaction_fee=0.0001
                )
        
        # Create and return vectorized environment
        try:
            return DummyVecEnv([_init])
        except Exception as e:
            print(f"Error creating vectorized environment: {str(e)}")
            # Create simplest possible environment
            return DummyVecEnv([lambda: ForexTradingEnv(df, self.lstm_model, initial_balance=10000.0)])
    
    def train(self, train_df: pd.DataFrame, total_timesteps: int = 100000) -> None:
        try:
            # Check data
            if train_df.empty:
                raise ValueError("Training data is empty!")
                
            # Check for NaN values and clean
            if train_df.isnull().values.any():
                print("Warning: NaN values found in training data. Filling with forward fill method...")
                train_df = train_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Check if all required technical indicators exist
            required_cols = ['RSI', 'MACD', 'Signal_Line', 'ATR', 'Upper_Band', 'Lower_Band', 'MA20']
            missing_cols = [col for col in required_cols if col not in train_df.columns]
            
            if missing_cols:
                print(f"Warning: Missing technical indicators: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    if col == 'RSI':
                        train_df[col] = 50  # Neutral RSI
                    elif col in ['MACD', 'Signal_Line']:
                        train_df[col] = 0  # Neutral MACD
                    elif col == 'ATR':
                        train_df[col] = train_df['close'] * 0.01  # 1% volatility
                    elif col in ['Upper_Band', 'Lower_Band']:
                        if 'MA20' in train_df.columns:
                            # If MA20 exists, use it to create bands
                            std = train_df['close'].rolling(window=20, min_periods=1).std().fillna(0)
                            if col == 'Upper_Band':
                                train_df[col] = train_df['MA20'] + (2 * std)
                            else:
                                train_df[col] = train_df['MA20'] - (2 * std)
                        else:
                            # Otherwise, use simple calculation
                            if col == 'Upper_Band':
                                train_df[col] = train_df['close'] * 1.02
                            else:
                                train_df[col] = train_df['close'] * 0.98
                    elif col == 'MA20':
                        train_df[col] = train_df['close'].rolling(window=20, min_periods=1).mean().fillna(train_df['close'])
            
            # Create a new environment for this training session
            self.env = self.create_env(train_df)
            
            # Initialize or load model
            if self.model is None:
                print("Creating new PPO model...")
                self.model = PPO(
                    "MlpPolicy",
                    self.env,  # Use the stored environment
                    verbose=1,
                    **self.model_params
                )
            else:
                # Önemli: Eğer model varsa, çevreyi güncelle
                print("Updating existing model with new environment...")
                self.model.set_env(self.env)  # Update environment
            
            # Make sure model has a valid environment before training
            if self.model.env is None:
                print("Environment not set correctly. Setting it now...")
                self.model.env = self.env
            
            # Train model
            print("Starting model training...")
            self.model.learn(total_timesteps=total_timesteps)
            print("Model training completed!")
            
        except Exception as e:
            import traceback
            print(f"Error during training: {str(e)}")
            traceback.print_exc()
            raise
    
    def predict(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts action for given observation.
        
        Args:
            obs: Observation, shape (15,) or (1, 15)
            
        Returns:
            action: Predicted action
            _: Model state (not used)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Check observation shape and adjust
        if isinstance(obs, torch.Tensor):
            obs = obs.numpy()
            
        if len(obs.shape) == 1:
            obs = obs.reshape(1, -1)
            
        if obs.shape[1] != 15:
            raise ValueError(f"Observation shape must be (1, 15), got {obs.shape}")
            
        return self.model.predict(obs)
    
    def save(self, path: str) -> None:
        """Saves model to file"""
        try:
            if self.model is not None:
                print(f"Saving RL model to {path}...")
                self.model.save(path)
                print(f"RL model saved successfully")
            else:
                print("Warning: No model to save!")
        except Exception as e:
            print(f"Error saving RL model: {str(e)}")
    
    def load(self, path: str) -> None:
        """Loads model from file"""
        try:
            print(f"Loading RL model from {path}...")
            # Make sure we have a valid environment
            if self.env is None:
                print("Creating environment for model loading...")
                self.env = self.create_env(self.env_params.get('df', pd.DataFrame()))
            
            # Load the model with the environment
            self.model = PPO.load(path, env=self.env)
            
            # Double check that the environment is properly set
            if self.model.env is None:
                print("Environment not set after loading. Setting it now...")
                self.model.set_env(self.env)
            
            # Ensure the model parameters are set
            if hasattr(self.model, 'learning_rate'):
                self.model_params['learning_rate'] = self.model.learning_rate
            
            print(f"RL model loaded successfully")
        except Exception as e:
            print(f"Error loading RL model: {str(e)}")
            # Create a new model if loading fails
            print("Creating new RL model instead...")
            if self.env is None:
                self.env = self.create_env(self.env_params.get('df', pd.DataFrame()))
            
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                **self.model_params
            ) 