import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import os

class LSTMPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Get output of last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Fully connected layer
        out = self.fc(last_time_step)
        return out
        
    def train_model(self, train_sequences, train_targets, val_sequences=None, val_targets=None,
                   epochs=100, batch_size=32, learning_rate=0.001):
        """Trains the model"""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            num_batches = len(train_sequences) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_sequences = train_sequences[start_idx:end_idx]
                batch_targets = train_targets[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = self(batch_sequences)
                loss = criterion(outputs.squeeze(), batch_targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / num_batches
            
            # Validation
            if val_sequences is not None and val_targets is not None:
                val_loss = self.evaluate(val_sequences, val_targets)
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}')
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered!")
                        break
            else:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}')
    
    def evaluate(self, val_sequences, val_targets):
        """Evaluates the model"""
        self.eval()
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            outputs = self(val_sequences)
            val_loss = criterion(outputs.squeeze(), val_targets)
            
        return val_loss.item()
    
    def predict(self, sequence):
        """Makes prediction for a single sequence"""
        self.eval()
        with torch.no_grad():
            prediction = self(sequence)
        return prediction
    
    def save_model(self, path='saved_models'):
        """Saves the model"""
        if not os.path.exists(path):
            os.makedirs(path)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{path}/lstm_model_{timestamp}.pth"
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }, filename)
        
        print(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename):
        """Loads a saved model"""
        checkpoint = torch.load(filename)
        
        model = cls(
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 