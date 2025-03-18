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
import logging
from utils.data_processor import DataProcessor

logger = logging.getLogger("TradingBot.RLModel")

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
    def __init__(self, lstm_models=None, env_params=None):
        """
        RL trader initialization
        
        Args:
            lstm_models: Dict of {timeframe: LSTMPredictor} with trained LSTM models for each timeframe
            env_params: Environment parameters including:
                - df: DataFrame with price data (dictionary containing dataframes for each timeframe)
                - window_size: Size of observation window
                - initial_balance: Starting balance
                - commission: Trading commission
        """
        self.lstm_models = lstm_models if lstm_models else {}
        self.timeframes = list(self.lstm_models.keys()) if lstm_models else []
        self.env_params = env_params
        self.models = {}  # Dictionary to store RL models for each timeframe
        
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
        
        # Initialize environments for each timeframe
        self.train_envs = {}
        self.val_envs = {}
        
        # Training metrics
        self.train_metrics = {tf: [] for tf in self.timeframes}
        self.val_metrics = {tf: [] for tf in self.timeframes}
        
        # Initialize models if data is provided
        if env_params and lstm_models:
            self._initialize_environments()
        
    def _initialize_environments(self):
        """Initialize training and validation environments for each timeframe"""
        for timeframe in self.timeframes:
            if timeframe not in self.env_params['df']:
                logger.warning(f"No data available for timeframe {timeframe}, skipping...")
                continue
                
            df = self.env_params['df'][timeframe]
            
            # Split data for training and validation
            train_size = int(len(df) * 0.8)  # 80% training, 20% validation
            self.train_envs[timeframe] = self.create_env(
                df=df[:train_size], 
                timeframe=timeframe
            )
            self.val_envs[timeframe] = self.create_env(
                df=df[train_size:], 
                timeframe=timeframe
            )
        
    def create_env(self, df: pd.DataFrame, timeframe: str) -> gym.Env:
        """
        Create trading environment for a specific timeframe
        
        Args:
            df: DataFrame with price data
            timeframe: Timeframe string (e.g., '5m', '15m', '1h')
            
        Returns:
            gym.Env: Trading environment
        """
        if timeframe not in self.lstm_models:
            raise ValueError(f"LSTM model for timeframe {timeframe} not found")
            
        lstm_model = self.lstm_models[timeframe]
        
        # Prepare LSTM predictions for the entire dataset
        lstm_predictions = []
        
        # Process data in chunks to avoid memory issues
        chunk_size = 1000
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            # Prepare data for LSTM
            lstm_data = self.prepare_lstm_data(chunk, timeframe)
            # Get LSTM predictions
            with torch.no_grad():
                pred = lstm_model(lstm_data)
                lstm_predictions.extend(pred.numpy())
        
        # Add LSTM predictions to the DataFrame
        df['lstm_prediction'] = lstm_predictions
        
        # Create environment
        env = ForexTradingEnv(
            df=df,
            lstm_model=lstm_model,
            initial_balance=self.env_params['initial_balance'],
            max_position_size=self.env_params.get('max_position_size', 1.0),
            transaction_fee=self.env_params['commission']
        )
        
        return env
    
    def prepare_lstm_data(self, df: pd.DataFrame, timeframe: str) -> torch.Tensor:
        """
        Prepare data for LSTM model
        
        Args:
            df: DataFrame with price data
            timeframe: Timeframe string (e.g., '5m', '15m', '1h')
            
        Returns:
            torch.Tensor: Data for LSTM model
        """
        # Implementation depends on data structure
        # This is a placeholder - actual implementation should match your LSTM data preparation
        return torch.tensor(df.values, dtype=torch.float32)
    
    def train(self, total_timesteps: int = 10000, eval_freq: int = 1000, verbose: bool = True):
        """
        Train RL models for each timeframe
        
        Args:
            total_timesteps: Total number of timesteps to train for each model
            eval_freq: Evaluation frequency (in timesteps)
            verbose: Whether to print training progress
        """
        for timeframe in self.timeframes:
            print(f"\n=== Training RL model for {timeframe} timeframe ===")
            
            if timeframe not in self.train_envs or timeframe not in self.val_envs:
                print(f"No environment for timeframe {timeframe}, skipping...")
                continue
                
            try:
                # Create and train model
                model = PPO('MlpPolicy', 
                           self.train_envs[timeframe], 
                           learning_rate=self.model_params['learning_rate'],
                           batch_size=self.model_params['batch_size'],
                           n_steps=self.model_params['n_steps'],
                           gamma=self.model_params['gamma'],
                           policy_kwargs=self.model_params['policy_kwargs'],
                           verbose=1)
                
                self.models[timeframe] = model
                
                # Train with evaluation
                timesteps_elapsed = 0
                
                while timesteps_elapsed < total_timesteps:
                    if verbose:
                        print(f"\nTraining {timeframe} model: {timesteps_elapsed}/{total_timesteps} timesteps")
                    
                    # Train for eval_freq timesteps
                    self.models[timeframe].learn(
                        total_timesteps=min(eval_freq, total_timesteps - timesteps_elapsed),
                        reset_num_timesteps=False
                    )
                    timesteps_elapsed += eval_freq
                    
                    # Evaluate on training environment
                    train_metrics = self.evaluate_model(self.train_envs[timeframe])
                    self.train_metrics[timeframe].append({
                        'timestep': timesteps_elapsed,
                        **train_metrics
                    })
                    
                    # Evaluate on validation environment
                    val_metrics = self.evaluate_model(self.val_envs[timeframe])
                    self.val_metrics[timeframe].append({
                        'timestep': timesteps_elapsed,
                        **val_metrics
                    })
                    
                    # Print progress
                    if verbose:
                        print(f"Training - Avg Return: {train_metrics['avg_return']:.2%}, Win Rate: {train_metrics['win_rate']:.2%}")
                        print(f"Validation - Avg Return: {val_metrics['avg_return']:.2%}, Win Rate: {val_metrics['win_rate']:.2%}")
                
                print(f"Training for {timeframe} completed!")
                
            except Exception as e:
                print(f"Error training {timeframe} model: {str(e)}")
                logger.error(f"Error training {timeframe} model: {str(e)}")
        
        print("\nAll models trained!")
    
    def predict_combined(self, states: dict) -> Tuple[int, dict]:
        """
        Predict action using all available timeframe models
        
        Args:
            states: Dict of {timeframe: state} for each timeframe
            
        Returns:
            action: Combined predicted action (0: Hold, 1: Buy, 2: Sell)
            details: Details of individual predictions and confidence
        """
        if not self.models:
            raise ValueError("No trained models available!")
        
        predictions = {}
        confidences = {}
        
        # Define timeframe weights - shorter timeframes are more reactive but noisier
        # Longer timeframes are more stable but may be delayed
        timeframe_weights = {
            '1m': 0.5,    # Very short-term (noisy)
            '5m': 0.8,    # Short-term
            '15m': 1.0,   # Medium-term (base weight)
            '30m': 1.2,   # Medium-long term
            '1h': 1.5,    # Long-term (more stable)
            '4h': 1.8,    # Very long-term
        }
        
        # Default weight if timeframe not in the weights dictionary
        default_weight = 1.0
        
        # Get predictions for each timeframe
        model_outputs = {}
        lstm_predictions = {}
        
        for timeframe, model in self.models.items():
            if timeframe not in states:
                logger.warning(f"No state available for timeframe {timeframe}, skipping...")
                continue
                
            state = states[timeframe]
            
            # Get LSTM prediction for current state
            lstm_model = self.lstm_models[timeframe]
            lstm_data = self.prepare_lstm_data(state, timeframe)
            
            try:
                with torch.no_grad():
                    lstm_pred = lstm_model(lstm_data)
                    lstm_predictions[timeframe] = lstm_pred.numpy()
                
                # Add LSTM prediction to state
                enhanced_state = np.append(state, lstm_pred.numpy())
                
                # Get raw RL output (before argmax)
                action_probs, _ = model.predict(enhanced_state, deterministic=False)
                model_outputs[timeframe] = action_probs
                
                # Record action (argmax of probabilities)
                action = np.argmax(action_probs)
                predictions[timeframe] = int(action)
                
                # Confidence is the probability of the chosen action
                action_confidence = action_probs[action]
                confidences[timeframe] = float(action_confidence)
            except Exception as e:
                logger.error(f"Error getting prediction for {timeframe}: {str(e)}")
                continue
        
        if not predictions:
            return 0, {"error": "No predictions available"}
        
        # Combine predictions with weighted voting
        votes = {0: 0.0, 1: 0.0, 2: 0.0}  # Hold, Buy, Sell
        
        for timeframe, action in predictions.items():
            # Get weight for this timeframe
            weight = timeframe_weights.get(timeframe, default_weight)
            
            # Multiply weight by the model's confidence
            adjusted_weight = weight * confidences[timeframe]
            
            votes[action] += adjusted_weight
        
        # Get action with highest vote
        max_vote = max(votes.values())
        winning_actions = [action for action, vote in votes.items() if vote == max_vote]
        
        # Calculate confidence as proportion of winning votes to total votes
        total_votes = sum(votes.values())
        confidence = max_vote / total_votes if total_votes > 0 else 0.0
        
        # If tie, prefer hold (0)
        if len(winning_actions) > 1 and 0 in winning_actions:
            final_action = 0
        else:
            final_action = winning_actions[0]
        
        details = {
            'individual_predictions': predictions,
            'confidences': confidences,
            'votes': votes,
            'total_votes': total_votes,
            'confidence': confidence,
            'final_action': final_action,
            'lstm_predictions': lstm_predictions,
            'timeframe_weights': {tf: timeframe_weights.get(tf, default_weight) for tf in predictions.keys()}
        }
        
        return final_action, details
    
    def predict(self, state: np.ndarray, timeframe: str = None) -> Tuple[int, float]:
        """
        Predict action for given state and timeframe
        
        Args:
            state: Current state
            timeframe: Timeframe to use for prediction (if None, use the first available)
            
        Returns:
            action: Predicted action (0: Hold, 1: Buy, 2: Sell)
            confidence: Model's confidence in the action
        """
        if not self.models:
            raise ValueError("No trained models available!")
        
        # If timeframe not specified, use the first available
        if timeframe is None:
            timeframe = next(iter(self.models.keys()))
        
        if timeframe not in self.models:
            raise ValueError(f"No model available for timeframe {timeframe}")
        
        model = self.models[timeframe]
        lstm_model = self.lstm_models[timeframe]
        
        # Get LSTM prediction for current state
        lstm_data = self.prepare_lstm_data(state, timeframe)
        with torch.no_grad():
            lstm_pred = lstm_model(lstm_data)
        
        # Add LSTM prediction to state
        enhanced_state = np.append(state, lstm_pred.numpy())
        
        # Get RL prediction
        action, _ = model.predict(enhanced_state, deterministic=True)
        
        return int(action), 1.0  # Confidence is always 1.0 for now
    
    def save(self, path_prefix: str) -> None:
        """
        Save all models and training metrics
        
        Args:
            path_prefix: Path prefix to save models
        """
        if not self.models:
            raise ValueError("No models to save")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
        
        # Save each model
        for timeframe, model in self.models.items():
            model_path = f"{path_prefix}_rl_{timeframe}.zip"
            model.save(model_path)
            print(f"RL model for {timeframe} saved to {model_path}")
            
            # Save metrics
            metrics_path = f"{path_prefix}_rl_{timeframe}_metrics.json"
            metrics = {
                'train_metrics': self.train_metrics[timeframe],
                'val_metrics': self.val_metrics[timeframe]
            }
            
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
    
    def load(self, path_prefix: str, timeframes: List[str]) -> None:
        """
        Load models from files
        
        Args:
            path_prefix: Path prefix to load models from
            timeframes: List of timeframes to load models for
        """
        self.models = {}
        
        for timeframe in timeframes:
            model_path = f"{path_prefix}_rl_{timeframe}.zip"
            
            try:
                model = PPO.load(model_path)
                self.models[timeframe] = model
                print(f"RL model for {timeframe} loaded from {model_path}")
                
                # Try to load metrics
                metrics_path = f"{path_prefix}_rl_{timeframe}_metrics.json"
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        self.train_metrics[timeframe] = metrics.get('train_metrics', [])
                        self.val_metrics[timeframe] = metrics.get('val_metrics', [])
                except FileNotFoundError:
                    print(f"No metrics file found for {timeframe}")
                
            except Exception as e:
                print(f"Error loading model for {timeframe}: {str(e)}")
    
    def evaluate_model(self, env, episodes=10, timeframe=None):
        """
        Evaluate model on environment
        
        Args:
            env: Environment to evaluate on
            episodes: Number of episodes to evaluate
            timeframe: Timeframe to use (if None, uses the first available)
            
        Returns:
            Dict with evaluation metrics
        """
        if timeframe is None:
            timeframe = next(iter(self.models.keys()))
            
        if timeframe not in self.models:
            raise ValueError(f"No model available for timeframe {timeframe}")
            
        returns = []
        wins = 0
        
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0
            initial_balance = env.balance
            
            while not done:
                action, _ = self.models[timeframe].predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_return += reward
            
            # Check if we made money
            if env.balance > initial_balance:
                wins += 1
                
            returns.append((env.balance - initial_balance) / initial_balance)
        
        return {
            'avg_return': sum(returns) / len(returns),
            'win_rate': wins / episodes
        } 