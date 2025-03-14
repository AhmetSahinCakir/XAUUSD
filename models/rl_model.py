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
import json

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
            shape=(16,),  # price data (5) + technical indicators (7) + account state (3) + lstm prediction (1)
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
        
        # LSTM prediction (1 feature)
        lstm_prediction = np.array([current_data['lstm_prediction']])
        
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
        normalized_lstm = safe_normalize(lstm_prediction)
        
        # Combine all data and flatten
        observation = np.concatenate([
            normalized_price,     # 5 features
            normalized_tech,      # 7 features
            account_state,       # 3 features
            normalized_lstm      # 1 feature
        ]).flatten()  # Total 16 features
        
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
        """
        RL trader initialization
        
        Args:
            lstm_model: Trained LSTM model for price predictions
            env_params: Environment parameters including:
                - df: DataFrame with price data
                - window_size: Size of observation window
                - initial_balance: Starting balance
                - commission: Trading commission
        """
        self.lstm_model = lstm_model
        self.model = None
        
        # Model parameters
        self.model_params = {
            'learning_rate': 0.0001,
            'batch_size': 64,
            'n_steps': 2048,
            'gamma': 0.99,
            'policy_kwargs': dict(
                net_arch=[256, 128, 64]  # Deeper network for more complex patterns
            )
        }
        
        # Split data for training and validation
        train_size = int(len(env_params['df']) * 0.8)  # 80% training, 20% validation
        self.train_df = env_params['df'][:train_size]
        self.val_df = env_params['df'][train_size:]
        
        # Create environments
        self.train_env = self.create_env(self.train_df)
        self.val_env = self.create_env(self.val_df)
        
        # Training metrics
        self.train_metrics = []
        self.val_metrics = []
        
    def create_env(self, df: pd.DataFrame) -> gym.Env:
        """
        Create trading environment
        
        Args:
            df: DataFrame with price data
            
        Returns:
            gym.Env: Trading environment
        """
        # Prepare LSTM predictions for the entire dataset
        lstm_predictions = []
        
        # Process data in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            # Prepare data for LSTM
            lstm_data = self.prepare_lstm_data(chunk)
            # Get LSTM predictions
            with torch.no_grad():
                pred = self.lstm_model(lstm_data)
                lstm_predictions.extend(pred.numpy())
        
        # Add LSTM predictions to the DataFrame
        df['lstm_prediction'] = lstm_predictions
        
        # Create environment with enhanced state space
        env = ForexTradingEnv(
            df=df,
            lstm_model=self.lstm_model,
            initial_balance=self.env_params['initial_balance'],
            max_position_size=self.env_params['max_position_size'],
            transaction_fee=self.env_params['commission']
        )
        
        return env
    
    def prepare_lstm_data(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Prepare data for LSTM model
        
        Args:
            df: DataFrame with price data
            
        Returns:
            torch.Tensor: Prepared data for LSTM
        """
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Normalize data
        df = normalize_data(df)
        
        # Convert to tensor
        data = torch.FloatTensor(df.values)
        
        return data
    
    def evaluate_model(self, env: gym.Env, num_episodes: int = 5) -> dict:
        """
        Evaluate model performance on given environment
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        total_rewards = []
        total_returns = []
        win_rate = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            initial_balance = env.balance
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
            
            # Calculate metrics
            total_rewards.append(episode_reward)
            episode_return = (env.balance - initial_balance) / initial_balance
            total_returns.append(episode_return)
            win_rate += 1 if episode_return > 0 else 0
        
        # Calculate average metrics
        avg_reward = np.mean(total_rewards)
        avg_return = np.mean(total_returns)
        win_rate = win_rate / num_episodes
        
        return {
            'avg_reward': avg_reward,
            'avg_return': avg_return,
            'win_rate': win_rate,
            'total_rewards': total_rewards,
            'total_returns': total_returns
        }
        
    def train(self, train_df: pd.DataFrame, total_timesteps: int = 100000, eval_freq: int = 10000) -> None:
        """
        Train the RL model
        
        Args:
            train_df: Training data
            total_timesteps: Total number of training steps
            eval_freq: Frequency of evaluation (in steps)
        """
        try:
            print("Creating PPO model...")
            self.model = PPO(
                "MlpPolicy",
                self.train_env,
                verbose=1,
                **self.model_params
            )
            
            print(f"Starting training for {total_timesteps:,} timesteps...")
            
            # Training loop with periodic evaluation
            timesteps_elapsed = 0
            while timesteps_elapsed < total_timesteps:
                # Train for eval_freq steps
                self.model.learn(
                    total_timesteps=min(eval_freq, total_timesteps - timesteps_elapsed),
                    reset_num_timesteps=False
                )
                timesteps_elapsed += eval_freq
                
                # Evaluate on training environment
                train_metrics = self.evaluate_model(self.train_env)
                self.train_metrics.append({
                    'timestep': timesteps_elapsed,
                    **train_metrics
                })
                
                # Evaluate on validation environment
                val_metrics = self.evaluate_model(self.val_env)
                self.val_metrics.append({
                    'timestep': timesteps_elapsed,
                    **val_metrics
                })
                
                # Print progress
                print(f"\nTimestep: {timesteps_elapsed:,}/{total_timesteps:,}")
                print(f"Training - Avg Return: {train_metrics['avg_return']:.2%}, Win Rate: {train_metrics['win_rate']:.2%}")
                print(f"Validation - Avg Return: {val_metrics['avg_return']:.2%}, Win Rate: {val_metrics['win_rate']:.2%}")
            
            print("\nTraining completed!")
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
    
    def predict(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Predict action for given state
        
        Args:
            state: Current state including LSTM prediction
            
        Returns:
            action: Predicted action (0: Hold, 1: Buy, 2: Sell)
            confidence: Model's confidence in the action
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get LSTM prediction for current state
        lstm_data = self.prepare_lstm_data(state)
        with torch.no_grad():
            lstm_pred = self.lstm_model(lstm_data)
        
        # Add LSTM prediction to state
        enhanced_state = np.append(state, lstm_pred.numpy())
        
        # Get RL prediction
        action, _ = self.model.predict(enhanced_state, deterministic=True)
        
        return action
    
    def save(self, path: str) -> None:
        """
        Save model and training metrics
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        # Save model
        self.model.save(path)
        
        # Save metrics
        metrics_path = path.replace('.zip', '_metrics.json')
        metrics = {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    def load(self, path: str) -> None:
        """Load model from file"""
        try:
            self.model = PPO.load(path)
            print(f"RL model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise 