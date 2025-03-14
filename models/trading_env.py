import gym
import pandas as pd
import numpy as np
from gym import spaces

class ForexTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, window_size: int = 60, initial_balance: float = 10000.0, commission: float = 0.0001):
        """
        Initialize Forex trading environment
        
        Args:
            df: DataFrame with price data and LSTM predictions
            window_size: Size of observation window
            initial_balance: Starting balance
            commission: Trading commission
        """
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        
        # Trading state
        self.balance = initial_balance
        self.position = 0  # -1: Short, 0: Hold, 1: Long
        self.position_price = 0
        self.current_step = window_size
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # Enhanced observation space including LSTM predictions
        n_features = len(df.columns)  # Price data + Technical indicators + LSTM predictions
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, n_features),
            dtype=np.float32
        )
        
    def reset(self):
        """Reset environment to initial state"""
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.current_step = self.window_size
        
        return self._get_observation()
        
    def step(self, action):
        """
        Take action in environment
        
        Args:
            action: Trading action (0: Hold, 1: Buy, 2: Sell)
            
        Returns:
            observation: Current state
            reward: Reward for action
            done: Whether episode is finished
            info: Additional information
        """
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, {}
            
        # Get current price data
        current_price = self.df.iloc[self.current_step]['close']
        next_price = self.df.iloc[self.current_step + 1]['close']
        lstm_pred = self.df.iloc[self.current_step]['lstm_prediction']
        
        # Calculate reward
        reward = self._calculate_reward(action, current_price, next_price, lstm_pred)
        
        # Update position
        self._update_position(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'position_price': self.position_price
        }
        
        return observation, reward, done, info
        
    def _get_observation(self):
        """Get current state observation"""
        # Get window of data
        observation = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        
        return observation.astype(np.float32)
        
    def _calculate_reward(self, action, current_price, next_price, lstm_pred):
        """
        Calculate reward for action
        
        The reward function now considers both:
        1. The actual price movement (realized P/L)
        2. The accuracy of the LSTM prediction
        """
        reward = 0
        
        # Calculate price change
        price_change = (next_price - current_price) / current_price
        
        # Calculate LSTM prediction accuracy
        pred_error = abs(next_price - lstm_pred) / current_price
        pred_accuracy = 1 - pred_error  # Higher accuracy = smaller error
        
        # Base reward on position and price movement
        if self.position == 1:  # Long position
            reward = price_change
        elif self.position == -1:  # Short position
            reward = -price_change
            
        # Modify reward based on action and LSTM prediction accuracy
        if action == 1:  # Buy
            if lstm_pred > current_price:  # LSTM predicts price increase
                reward *= (1 + pred_accuracy)  # Boost reward if prediction was accurate
            else:
                reward *= (1 - pred_accuracy)  # Reduce reward if prediction was wrong
        elif action == 2:  # Sell
            if lstm_pred < current_price:  # LSTM predicts price decrease
                reward *= (1 + pred_accuracy)  # Boost reward if prediction was accurate
            else:
                reward *= (1 - pred_accuracy)  # Reduce reward if prediction was wrong
                
        # Apply transaction cost penalty for trades
        if action != 0:  # If trade was made
            reward -= self.commission
            
        return reward
        
    def _update_position(self, action, current_price):
        """Update position based on action"""
        if action == 1:  # Buy
            if self.position == -1:  # Close short position
                self.balance += (self.position_price - current_price) * abs(self.position)
            self.position = 1
            self.position_price = current_price
        elif action == 2:  # Sell
            if self.position == 1:  # Close long position
                self.balance += (current_price - self.position_price) * abs(self.position)
            self.position = -1
            self.position_price = current_price
            
    def render(self, mode='human'):
        """Render environment state"""
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Position: {self.position}')
        if self.position != 0:
            print(f'Position Price: ${self.position_price:.2f}')
        print('-' * 30) 