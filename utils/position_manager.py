import logging
import MetaTrader5 as mt5
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from config import TRADING_CONFIG

logger = logging.getLogger("TradingBot.PositionManager")

class PositionManager:
    """Gelişmiş pozisyon yönetimi sınıfı"""
    
    def __init__(self, mt5_connector):
        self.mt5 = mt5_connector
        self.positions: Dict[int, dict] = {}  # ticket -> position_info
        self.partial_tp_levels = TRADING_CONFIG['partial_tp_levels']
        self.trailing_stop = TRADING_CONFIG['trailing_stop']
        self.trailing_stop_factor = TRADING_CONFIG['trailing_stop_factor']
    
    def open_position(self, symbol: str, order_type: int, volume: float,
                     price: float, sl: float, tp: float, comment: str = "") -> Optional[int]:
        """
        Yeni pozisyon aç
        
        Parametreler:
        - symbol: İşlem sembolü
        - order_type: İşlem tipi (ALIM/SATIM)
        - volume: Lot büyüklüğü
        - price: Giriş fiyatı
        - sl: Stop loss seviyesi
        - tp: Take profit seviyesi
        - comment: İşlem notu
        
        Dönüş:
        - Pozisyon ticket numarası veya None (başarısız durumda)
        """
        try:
            # Kısmi kar alma seviyeleri için birden fazla pozisyon aç
            if TRADING_CONFIG['partial_tp_enabled']:
                ticket_list = []
                
                for level in self.partial_tp_levels:
                    partial_volume = volume * (level['percentage'] / 100)
                    partial_tp = price + (tp - price) * level['at_price_factor'] \
                               if order_type == mt5.ORDER_TYPE_BUY \
                               else price - (price - tp) * level['at_price_factor']
                    
                    result = self.mt5.place_order(
                        symbol=symbol,
                        order_type=order_type,
                        volume=partial_volume,
                        price=price,
                        sl=sl,
                        tp=partial_tp,
                        comment=f"{comment} (TP{level['at_price_factor']}R)"
                    )
                    
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        ticket_list.append(result.order)
                        self.positions[result.order] = {
                            'ticket': result.order,
                            'symbol': symbol,
                            'type': order_type,
                            'volume': partial_volume,
                            'price': price,
                            'sl': sl,
                            'tp': partial_tp,
                            'comment': comment,
                            'open_time': datetime.now()
                        }
                    else:
                        logger.error(f"Kısmi pozisyon açma hatası: {result.comment if result else 'Bilinmeyen hata'}")
                
                return ticket_list[0] if ticket_list else None
                
            else:
                # Tek pozisyon aç
                result = self.mt5.place_order(
                    symbol=symbol,
                    order_type=order_type,
                    volume=volume,
                    price=price,
                    sl=sl,
                    tp=tp,
                    comment=comment
                )
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.positions[result.order] = {
                        'ticket': result.order,
                        'symbol': symbol,
                        'type': order_type,
                        'volume': volume,
                        'price': price,
                        'sl': sl,
                        'tp': tp,
                        'comment': comment,
                        'open_time': datetime.now()
                    }
                    return result.order
                else:
                    logger.error(f"Pozisyon açma hatası: {result.comment if result else 'Bilinmeyen hata'}")
                    return None
                    
        except Exception as e:
            logger.error(f"Pozisyon açma işleminde hata: {str(e)}")
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
                        logger.info(f"Trailing stop güncellendi - Ticket: {ticket}, "
                                  f"Yeni SL: {new_sl:.2f}")
                    
        except Exception as e:
            logger.error(f"Trailing stop güncellemesinde hata: {str(e)}")
    
    def close_position(self, ticket: int) -> bool:
        """Pozisyonu kapat"""
        try:
            result = self.mt5.close_position(ticket)
            if result:
                if ticket in self.positions:
                    del self.positions[ticket]
                return True
            return False
        except Exception as e:
            logger.error(f"Pozisyon kapatma hatası: {str(e)}")
            return False
    
    def close_all_positions(self) -> bool:
        """Tüm pozisyonları kapat"""
        try:
            success = True
            for ticket in list(self.positions.keys()):
                if not self.close_position(ticket):
                    success = False
            return success
        except Exception as e:
            logger.error(f"Tüm pozisyonları kapatma hatası: {str(e)}")
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