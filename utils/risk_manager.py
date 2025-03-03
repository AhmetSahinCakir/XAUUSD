class RiskManager:
    def __init__(self, initial_balance, risk_per_trade=1, max_daily_loss=5):
        """
        Initializes the risk management class
        
        Args:
            initial_balance (float): Initial balance
            risk_per_trade (float): Risk percentage per trade (default: 1%)
            max_daily_loss (float): Maximum daily loss percentage (default: 5%)
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.max_daily_loss = max_daily_loss
        
        self.daily_trades = []
        self.daily_pnl = 0
        
    def calculate_position_size(self, entry_price, stop_loss):
        """
        Calculates position size according to risk management
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss level
            
        Returns:
            float: Lot size
        """
        try:
            # Entry ve stop loss değerlerini kontrol et
            if entry_price <= 0 or stop_loss <= 0:
                print(f"Invalid entry ({entry_price}) or stop loss ({stop_loss}) values")
                return 0.01  # Minimum lot
                
            # Fiyat farkını hesapla
            risk_amount = abs(entry_price - stop_loss)
            
            if risk_amount <= 0:
                print("Risk amount cannot be zero or negative")
                risk_amount = entry_price * 0.01  # Varsayılan olarak %1 risk
                
            # Hesap riskini hesapla
            account_risk = self.current_balance * (self.risk_per_trade / 100)
            
            # Lot büyüklüğünü hesapla (1 lot = 100 oz altın)
            lot_size = account_risk / (risk_amount * 100)
            
            # Lot büyüklüğünü 0.01'in katları olarak yuvarla
            lot_size = round(lot_size, 2)
            
            # Lot büyüklüğü sınırları kontrol et (min 0.01, max 1.0)
            lot_size = max(0.01, min(lot_size, 1.0))
            
            return lot_size
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            return 0.01  # Hata durumunda minimum lot
        
    def can_trade(self):
        """
        Checks if new trades can be opened
        
        Returns:
            bool: Can trade?
        """
        # Check daily loss limit
        if abs(self.daily_pnl) >= (self.initial_balance * self.max_daily_loss / 100):
            return False
            
        return True
        
    def update_balance(self, pnl):
        """
        Updates balance after trade
        
        Args:
            pnl (float): Profit/Loss amount
        """
        self.current_balance += pnl
        self.daily_pnl += pnl
        self.daily_trades.append(pnl)
        
    def calculate_stop_loss(self, entry_price, atr_value, position_type="BUY"):
        """
        Calculates stop loss level based on ATR
        
        Args:
            entry_price (float): Entry price
            atr_value (float): ATR value
            position_type (str): Position type ("BUY" or "SELL")
            
        Returns:
            float: Stop loss level
        """
        atr_multiplier = 1.5
        
        if position_type == "BUY":
            stop_loss = entry_price - (atr_value * atr_multiplier)
        else:
            stop_loss = entry_price + (atr_value * atr_multiplier)
            
        return round(stop_loss, 2)
        
    def calculate_take_profit(self, entry_price, stop_loss, risk_reward_ratio=2):
        """
        Calculates take profit level based on Risk/Reward ratio
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss level
            risk_reward_ratio (float): Risk/Reward ratio
            
        Returns:
            float: Take profit level
        """
        risk = abs(entry_price - stop_loss)
        take_profit = entry_price + (risk * risk_reward_ratio) if stop_loss < entry_price else entry_price - (risk * risk_reward_ratio)
        
        return round(take_profit, 2)
        
    def reset_daily_stats(self):
        """Resets daily statistics"""
        self.daily_trades = []
        self.daily_pnl = 0 