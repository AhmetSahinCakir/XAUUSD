import logging
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from config.config import TRADING_CONFIG
from .logger import print_info, print_warning, print_error, print_success

logger = logging.getLogger("TradingBot.PositionManager")

class PositionManager:
    """Gelişmiş pozisyon yönetimi sınıfı"""
    
    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.positions: Dict[int, dict] = {}  # ticket -> position_info
        self.partial_tp_levels = TRADING_CONFIG['partial_tp_levels']
        self.trailing_stop = TRADING_CONFIG['trailing_stop']
        self.trailing_stop_factor = TRADING_CONFIG['trailing_stop_factor']
        print_info(
            "Pozisyon yöneticisi başlatıldı",
            "Position manager initialized"
        )
    
    def open_position(self, symbol: str, order_type: str, volume: float, price: float, sl: float, tp: float, comment: str = ""):
        """Open a new position"""
        try:
            # Validate inputs
            if not all([symbol, order_type, volume, price]):
                print_error(
                    "Geçersiz işlem parametreleri!",
                    "Invalid trade parameters!"
                )
                return None
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type.upper() == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = self.mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print_error(
                    f"İşlem açılamadı! Hata: {result.comment}",
                    f"Failed to open position! Error: {result.comment}"
                )
                return None
            
            # Store position info
            position_info = {
                "ticket": result.order,
                "symbol": symbol,
                "type": order_type,
                "volume": volume,
                "price": price,
                "sl": sl,
                "tp": tp,
                "comment": comment
            }
            self.positions[result.order] = position_info
            
            print_success(
                f"{symbol} {order_type} pozisyonu açıldı. Lot: {volume}, Fiyat: {price}, SL: {sl}, TP: {tp}",
                f"Opened {order_type} position on {symbol}. Volume: {volume}, Price: {price}, SL: {sl}, TP: {tp}"
            )
            return result.order
            
        except Exception as e:
            print_error(
                f"Pozisyon açma hatası: {str(e)}",
                f"Error opening position: {str(e)}"
            )
            return None
    
    def update_trailing_stops(self):
        """Trailing stop seviyelerini güncelle"""
        if not self.trailing_stop:
            return
            
        try:
            # Tüm açık pozisyonları kontrol et
            for ticket, pos_info in self.positions.items():
                position = self.mt5.get_position(ticket)
                if not position:
                    continue
                
                current_price = self.mt5.get_current_price(pos_info['symbol'])
                if not current_price:
                    continue
                
                # ATR değerini al
                atr = self._get_atr(pos_info['symbol'])
                if not atr:
                    continue
                
                new_sl = None
                
                if pos_info['type'] == mt5.ORDER_TYPE_BUY:
                    trailing_level = current_price - (atr * self.trailing_stop_factor)
                    if trailing_level > pos_info['sl'] and trailing_level < current_price:
                        new_sl = trailing_level
                else:  # SELL
                    trailing_level = current_price + (atr * self.trailing_stop_factor)
                    if trailing_level < pos_info['sl'] and trailing_level > current_price:
                        new_sl = trailing_level
                
                if new_sl:
                    result = self.mt5.modify_position(
                        ticket,
                        sl=new_sl,
                        tp=pos_info['tp']
                    )
                    
                    if result:
                        self.positions[ticket]['sl'] = new_sl
                        print_info(f"Trailing stop güncellendi - Ticket: {ticket}, Yeni SL: {new_sl:.2f}")
                    
        except Exception as e:
            print_error(f"Trailing stop güncellemesinde hata: {str(e)}")
    
    def close_position(self, ticket: int, partial: float = None):
        """Close a position"""
        try:
            position = self.positions.get(ticket)
            if not position:
                print_error(
                    f"Pozisyon bulunamadı! Ticket: {ticket}",
                    f"Position not found! Ticket: {ticket}"
                )
                return False
            
            # Calculate volume to close
            volume = position["volume"] if partial is None else min(partial, position["volume"])
            
            # Create close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position["symbol"],
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL if position["type"].upper() == "BUY" else mt5.ORDER_TYPE_BUY,
                "position": ticket,
                "price": mt5.symbol_info_tick(position["symbol"]).bid,
                "deviation": 20,
                "magic": 234000,
                "comment": "close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = self.mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print_error(
                    f"Pozisyon kapatılamadı! Hata: {result.comment}",
                    f"Failed to close position! Error: {result.comment}"
                )
                return False
            
            # Update position info
            if partial is None or volume >= position["volume"]:
                del self.positions[ticket]
                print_success(
                    f"Pozisyon kapatıldı. Ticket: {ticket}",
                    f"Position closed. Ticket: {ticket}"
                )
            else:
                position["volume"] -= volume
                print_success(
                    f"Pozisyon kısmi kapatıldı. Ticket: {ticket}, Kapatılan: {volume}, Kalan: {position['volume']}",
                    f"Position partially closed. Ticket: {ticket}, Closed: {volume}, Remaining: {position['volume']}"
                )
            
            return True
            
        except Exception as e:
            print_error(
                f"Pozisyon kapatma hatası: {str(e)}",
                f"Error closing position: {str(e)}"
            )
            return False
    
    def close_all_positions(self) -> bool:
        """Tüm pozisyonları kapat"""
        try:
            success = True
            total_profit = 0.0
            closed_count = 0
            for ticket in list(self.positions.keys()):
                position = self.mt5.get_position(ticket)
                if position:
                    if self.close_position(ticket):
                        total_profit += position.profit
                        closed_count += 1
                    else:
                        success = False
            
            if closed_count > 0:
                print_success(f"{closed_count} pozisyon kapatıldı. Toplam Kar/Zarar: ${total_profit:.2f}")
            return success
        except Exception as e:
            print_error(f"Tüm pozisyonları kapatma hatası: {str(e)}")
            return False
    
    def get_position_info(self, ticket: int) -> Optional[dict]:
        """Pozisyon bilgilerini getir"""
        return self.positions.get(ticket)
    
    def get_all_positions(self) -> List[dict]:
        """Tüm pozisyon bilgilerini getir"""
        return list(self.positions.values())
    
    def calculate_total_profit(self) -> float:
        """Tüm pozisyonların toplam karını hesapla"""
        total_profit = 0.0
        for ticket in self.positions:
            position = self.mt5.get_position(ticket)
            if position:
                total_profit += position.profit
        return total_profit
    
    def _get_atr(self, symbol: str, period: int = 14, timeframe: str = '1h') -> Optional[float]:
        """ATR değerini hesapla"""
        try:
            data = self.mt5.get_historical_data(symbol, timeframe, num_candles=period+1)
            if data is None or len(data) < period:
                return None
            
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return float(atr)
            
        except Exception as e:
            logger.error(f"ATR hesaplama hatası: {str(e)}")
            return None
    
    def get_open_positions(self) -> List[dict]:
        """Get all open positions"""
        try:
            positions = []
            for ticket, pos_info in self.positions.items():
                # Check if position still exists in MT5
                mt5_position = self.mt5.get_position(ticket)
                if mt5_position:
                    positions.append(pos_info)
            return positions
        except Exception as e:
            print_error(
                f"Açık pozisyonlar alınırken hata: {str(e)}",
                f"Error getting open positions: {str(e)}"
            )
            return [] 