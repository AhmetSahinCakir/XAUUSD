import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import math
import time
import logging
from .logger import (
    print_info,
    print_warning,
    print_error,
    print_success
)

logger = logging.getLogger(__name__)

class MT5Connector:
    """
    MetaTrader 5 bağlantı sınıfı
    MT5 API üzerinden piyasa verilerine erişim ve ticaret işlemleri için kullanılır
    """
    
    def __init__(self, login=None, password=None, server=None, timeout=60000, max_retries=3):
        """
        MT5 bağlantısını başlatır
        
        Parametreler:
        - login: MT5 hesap numarası (opsiyonel)
        - password: MT5 hesap şifresi (opsiyonel)
        - server: MT5 sunucu adı (opsiyonel)
        - timeout: Bağlantı zaman aşımı (milisaniye)
        - max_retries: Maksimum yeniden deneme sayısı
        """
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.max_retries = max_retries
        self.connected = False
        self.retry_count = 0
        self.last_error = None
        self.connect()
    
    def connect(self):
        """Connect to MetaTrader 5"""
        try:
            # Önceki bağlantıyı temizle
            if mt5.initialize():
                mt5.shutdown()
            
            # MT5'i başlat
            if not mt5.initialize():
                self.last_error = mt5.last_error()
                print_error(
                    "MT5 başlatılamadı!",
                    "Failed to initialize MT5!",
                    f"Hata: {self.last_error}",
                    f"Error: {self.last_error}"
                )
                return self._handle_connection_error()
            
            # Giriş yap
            if self.login and self.password and self.server:
                if not mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server
                ):
                    self.last_error = mt5.last_error()
                    print_error(
                        "MT5 giriş başarısız!",
                        "MT5 login failed!",
                        f"Hata: {self.last_error}",
                        f"Error: {self.last_error}"
                    )
                    return self._handle_connection_error()

            # Hesap bilgilerini kontrol et
            account_info = mt5.account_info()
            if account_info is None:
                self.last_error = mt5.last_error()
                print_error(
                    "Hesap bilgileri alınamadı!",
                    "Failed to get account info!",
                    f"Hata: {self.last_error}",
                    f"Error: {self.last_error}"
                )
                return self._handle_connection_error()

            self.connected = True
            self.retry_count = 0
            print_success(
                f"MT5 bağlantısı başarılı! Hesap: {account_info.login}, Sunucu: {account_info.server}",
                f"MT5 connection successful! Account: {account_info.login}, Server: {account_info.server}"
            )
            return True

        except Exception as e:
            self.last_error = str(e)
            print_error(
                f"MT5 bağlantı hatası: {str(e)}",
                f"MT5 connection error: {str(e)}"
            )
            return self._handle_connection_error()
    
    def _handle_connection_error(self):
        """Bağlantı hatalarını yönet"""
        self.retry_count += 1
        if self.retry_count < self.max_retries:
            print_warning(
                f"Bağlantı yeniden deneniyor ({self.retry_count}/{self.max_retries})...",
                f"Retrying connection ({self.retry_count}/{self.max_retries})..."
            )
            time.sleep(2 ** self.retry_count)  # Exponential backoff
            return self.connect()
        else:
            print_error(
                f"Maksimum yeniden deneme sayısına ulaşıldı ({self.max_retries})",
                f"Maximum retry count reached ({self.max_retries})"
            )
            return False
    
    def ensure_connected(self):
        """Bağlantının aktif olduğundan emin ol"""
        if not self.connected or not mt5.terminal_info():
            print_warning(
                "MT5 bağlantısı kopmuş, yeniden bağlanılıyor...",
                "MT5 connection lost, reconnecting..."
            )
            self.retry_count = 0
            return self.connect()
        return True
    
    def disconnect(self):
        """MT5 bağlantısını güvenli bir şekilde sonlandırır"""
        try:
            if not self.connected:
                return True  # Zaten bağlı değil
                
            # Açık pozisyonları kontrol et
            positions = mt5.positions_get()
            if positions:
                print_warning(f"{len(positions)} açık pozisyon var.")
            
            # Bekleyen emirleri kontrol et
            orders = mt5.orders_get()
            if orders:
                print_warning(f"{len(orders)} bekleyen emir var.")
            
            # MT5 bağlantısını kapat
            mt5.shutdown()
            self.connected = False
            print_info("MT5 bağlantısı güvenli bir şekilde kapatıldı.")
            return True
            
        except Exception as e:
            print_error(f"MT5 bağlantısı kapatılırken hata: {str(e)}")
            return False
    
    def get_account_info(self):
        """Hesap bilgilerini döndürür"""
        if not self.connected and not self.connect():
            print_error("MT5 bağlantısı kurulamadı, hesap bilgisi alınamıyor")
            return None
        
        try:
            return mt5.account_info()
        except Exception as e:
            print_error(f"Hesap bilgisi alınırken hata: {str(e)}")
            return None
    
    def symbol_info(self, symbol):
        """
        Belirtilen sembol için bilgileri döndürür
        
        Parametreler:
        - symbol: İşlem yapılacak sembol (ör. "XAUUSD")
        """
        if not self.connected and not self.connect():
            print_error(f"MT5 bağlantısı kurulamadı, {symbol} bilgisi alınamıyor")
            return None
        
        try:
            # Sembol bilgisini al
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print_error(f"'{symbol}' sembolü bulunamadı, sembol listede olduğundan emin olun")
                return None
            
            # Sembol görünür değilse görünür yap
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    print_error(f"'{symbol}' sembolü görünür yapılamadı, emir verilemedi")
                    return False
            
            return symbol_info
        except Exception as e:
            print_error(f"Sembol bilgisi alınırken hata: {str(e)}")
            return None
    
    def get_historical_data(self, symbol, timeframe, num_candles=500, start_date=None, end_date=None):
        """
        Belirtilen sembol ve zaman dilimi için tarihsel mum verilerini alır
        
        Parametreler:
        - symbol: İşlem yapılacak sembol (ör. "XAUUSD")
        - timeframe: Zaman dilimi (ör. "1h", "4h", "1d")
        - num_candles: Alınacak mum sayısı (varsayılan: 500)
        - start_date: Başlangıç tarihi (datetime nesnesi, opsiyonel)
        - end_date: Bitiş tarihi (datetime nesnesi, opsiyonel)
        """
        if not self.ensure_connected():
            print_error(f"MT5 bağlantısı kurulamadı, {symbol} için tarihsel veri alınamıyor")
            return None
            
        # Zaman dilimi çeviricisi
        tf_dict = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "1W": mt5.TIMEFRAME_W1,
            "1M": mt5.TIMEFRAME_MN1
        }
        
        if timeframe not in tf_dict:
            print_error(f"Geçersiz zaman dilimi: {timeframe}. Desteklenen değerler: {list(tf_dict.keys())}")
            return None
        
        mt5_timeframe = tf_dict[timeframe]
        
        try:
            # Sembol bilgisini kontrol et
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print_error(f"'{symbol}' sembolü bulunamadı")
                return None
            
            # Sembol seçili değilse seç
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    print_error(f"'{symbol}' sembolü seçilemedi")
                    return None
            
            # Veri alımı parametrelerini hazırla
            if start_date and end_date:
                rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            else:
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_candles)
            
            if rates is None or len(rates) == 0:
                print_error(f"'{symbol}' için veri alınamadı")
                return None
            
            # DataFrame'e dönüştür
            df = pd.DataFrame(rates)
            
            # Zaman dönüşümü ve index ayarı
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df.set_index('time', inplace=True)
            
            # Sütun isimlerini düzenle
            df.columns = [col.lower() for col in df.columns]
            
            print_success(
                f"{symbol} {timeframe} verisi alındı: {len(df)} mum",
                f"Retrieved {symbol} {timeframe} data: {len(df)} candles"
            )
            
            return df
            
        except Exception as e:
            print_error(f"Veri alımı sırasında hata: {str(e)}")
            return None
    
    def get_open_positions(self, symbol=None):
        """
        Açık pozisyonları döndürür
        
        Parametreler:
        - symbol: İsteğe bağlı, belirli bir sembol için açık pozisyonları filtrelemek için
        """
        if not self.connected and not self.connect():
            print_error("MT5 bağlantısı kurulamadı, açık pozisyonlar alınamıyor")
            return None
        
        try:
            # Açık pozisyonları al
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None or len(positions) == 0:
                return []
            
            # Pozisyonları DataFrame'e dönüştür
            df = pd.DataFrame(list(positions), columns=positions[0]._asdict().keys())
            return df
        
        except Exception as e:
            print_error(f"Açık pozisyonlar alınırken hata: {str(e)}")
            return None
    
    def get_orders(self, symbol=None):
        """
        Bekleyen emirleri döndürür
        
        Parametreler:
        - symbol: İsteğe bağlı, belirli bir sembol için bekleyen emirleri filtrelemek için
        """
        if not self.connected and not self.connect():
            print_error("MT5 bağlantısı kurulamadı, bekleyen emirler alınamıyor")
            return None
        
        try:
            # Bekleyen emirleri al
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
            
            if orders is None or len(orders) == 0:
                return []
            
            # Emirleri DataFrame'e dönüştür
            df = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())
            return df
        
        except Exception as e:
            print_error(f"Bekleyen emirler alınırken hata: {str(e)}")
            return None
    
    def place_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment="MT5 Bot Trade"):
        """
        Emir yerleştirir
        
        Parametreler:
        - symbol: İşlem yapılacak sembol (ör. "XAUUSD")
        - order_type: İşlem türü ("BUY" veya "SELL")
        - volume: İşlem hacmi (lot)
        - price: İşlem fiyatı (piyasa emri için None olabilir)
        - sl: Stop Loss fiyatı (opsiyonel)
        - tp: Take Profit fiyatı (opsiyonel)
        - comment: İşlem için açıklama (opsiyonel)
        
        Başarı durumunda True, başarısızlık durumunda False döndürür
        """
        # Bağlantı kontrolü
        if not self.connected and not self.connect():
            print_error("MT5 bağlantısı kurulamadı, emir verilemedi")
            return False
        
        try:
            # Sembol bilgilerini al
            symbol_info = self.symbol_info(symbol)
            
            if symbol_info is None:
                print_error(f"'{symbol}' sembolü bulunamadı, emir verilemedi")
                return False
            
            if not symbol_info.visible:
                print_warning(f"'{symbol}' sembolü görünür değil, görünür yapılmaya çalışılıyor...")
                if not mt5.symbol_select(symbol, True):
                    print_error(f"'{symbol}' sembolü görünür yapılamadı, emir verilemedi")
                    return False
            
            # Emir tipi ayarları
            action = None
            if order_type.upper() == "BUY":
                action = mt5.ORDER_TYPE_BUY
                price_type = mt5.SYMBOL_TRADE_EXECUTION_MARKET
            elif order_type.upper() == "SELL":
                action = mt5.ORDER_TYPE_SELL
                price_type = mt5.SYMBOL_TRADE_EXECUTION_MARKET
            else:
                print_error(f"Geçersiz emir tipi: {order_type}, sadece 'BUY' veya 'SELL' destekleniyor")
                return False
            
            # Doldurma politikası ayarları
            # MT5'te farklı semboller farklı doldurma politikalarını destekleyebilir
            # MT5 sürüm farkları nedeniyle uyumlu bir doldurma tipi kullanıyoruz
            filling_type = mt5.ORDER_FILLING_RETURN
            
            # Not: Bazı MT5 sürümlerinde SYMBOL_FILLING_FOK ve SYMBOL_FILLING_IOC özellikleri bulunmuyor
            # Bu nedenle doğrudan ORDER_FILLING_RETURN kullanıyoruz
            # Daha sonra burada bir try-except ile farklı sürümleri destekleyen kod eklenebilir
            
            # Mevcut fiyatı al (eğer piyasa emri kullanıyorsak)
            if price is None:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print_error(f"'{symbol}' için fiyat bilgisi alınamadı, emir verilemedi")
                    return False
                
                if action == mt5.ORDER_TYPE_BUY:
                    price = tick.ask
                else:  # SELL
                    price = tick.bid
            
            # Lot büyüklüğünü doğrula
            volume = float(volume)
            # Lot değerini sembolün özellikleriyle uyumlu olacak şekilde yuvarla
            volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
            
            # Minimum lot kontrolü
            if volume < symbol_info.volume_min:
                print_warning(f"Uyarı: Hacim ({volume}) minimum değerin altında, {symbol_info.volume_min} değerine ayarlandı")
                volume = symbol_info.volume_min
            # Maksimum lot kontrolü
            elif volume > symbol_info.volume_max:
                print_warning(f"Uyarı: Hacim ({volume}) maksimum değerin üstünde, {symbol_info.volume_max} değerine ayarlandı")
                volume = symbol_info.volume_max
            
            # Fiyatı sembolün adım büyüklüğüne göre yuvarla
            price = round(price / symbol_info.trade_tick_size) * symbol_info.trade_tick_size
            
            # SL/TP değerlerini yuvarla
            if sl is not None:
                sl = round(sl / symbol_info.trade_tick_size) * symbol_info.trade_tick_size
            
            if tp is not None:
                tp = round(tp / symbol_info.trade_tick_size) * symbol_info.trade_tick_size
            
            # İşlem talebini hazırla
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": action,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 10,  # fiyat sapması (pip)
                "magic": 12345,   # Magic number - bot kimliği için
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_type
            }
            
            # Emir detaylarını yazdır
            print_info("-" * 50)
            print_info("Emir Detayları:")
            print_info(f"Sembol: {symbol}")
            print_info(f"İşlem Tipi: {order_type}")
            print_info(f"Hacim: {volume}")
            print_info(f"Fiyat: {price}")
            if sl is not None:
                print_info(f"Stop Loss: {sl}")
            if tp is not None:
                print_info(f"Take Profit: {tp}")
            print_info(f"Doldurma Tipi: {filling_type}")
            print_info("-" * 50)
            
            # Emri gönder
            result = mt5.order_send(request)
            
            # Sonucu kontrol et
            if result is None:
                print_error(f"Emir gönderilemedi, MT5 hatası: {mt5.last_error()}")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                # Hata kodlarını kontrol et ve özel mesajlar göster
                error_messages = {
                    mt5.TRADE_RETCODE_REQUOTE: "Fiyat değişti, yeniden fiyatlandırma gerekiyor",
                    mt5.TRADE_RETCODE_REJECT: "İstek reddedildi",
                    mt5.TRADE_RETCODE_CANCEL: "İstek işlemci tarafından iptal edildi",
                    mt5.TRADE_RETCODE_INVALID: "Geçersiz istek",
                    mt5.TRADE_RETCODE_TIMEOUT: "İstek zaman aşımına uğradı",
                    mt5.TRADE_RETCODE_INVALID_VOLUME: "Geçersiz hacim",
                    mt5.TRADE_RETCODE_INVALID_PRICE: "Geçersiz fiyat",
                    mt5.TRADE_RETCODE_INVALID_STOPS: "Geçersiz stop seviyeleri",
                    mt5.TRADE_RETCODE_TRADE_DISABLED: "İşlem devre dışı bırakıldı",
                    mt5.TRADE_RETCODE_MARKET_CLOSED: "Piyasa kapalı",
                    mt5.TRADE_RETCODE_NO_MONEY: "İşlem için yeterli para yok",
                    mt5.TRADE_RETCODE_PRICE_CHANGED: "Fiyat değişti",
                    mt5.TRADE_RETCODE_PRICE_OFF: "Kotasyon yok",
                    mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Geçersiz emir sona erme tarihi",
                    mt5.TRADE_RETCODE_ORDER_CHANGED: "Emir değiştirildi",
                    mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Çok fazla istek",
                    mt5.TRADE_RETCODE_TRADE_DISABLED: "Hesapta işlem yapma devre dışı",
                    mt5.TRADE_RETCODE_FROZEN: "Sipariş/pozisyon donduruldu",
                    mt5.TRADE_RETCODE_INVALID_FILL: "Geçersiz doldurma türü",
                    mt5.TRADE_RETCODE_CONNECTION: "Sunucu ile bağlantı yok",
                    mt5.TRADE_RETCODE_DONE_PARTIAL: "İstek kısmen tamamlandı",
                    mt5.TRADE_RETCODE_LIMIT_ORDERS: "Bekleyen emir sayısı sınırı aşıldı",
                    mt5.TRADE_RETCODE_LIMIT_VOLUME: "İşlem hacim sınırı aşıldı"
                }
                
                error_message = error_messages.get(result.retcode, f"Bilinmeyen hata kodu: {result.retcode}")
                print_error(f"Emir başarısız oldu: {error_message}")
                print_info(f"Sonuç detayları: {result}")
                
                # Eğer fiyat geçersizse veya doldurma politikası hatalıysa, farklı bir politika deneyin
                if result.retcode in [mt5.TRADE_RETCODE_INVALID_FILL, mt5.TRADE_RETCODE_INVALID_PRICE]:
                    print_warning("Farklı bir doldurma politikası denenecek...")
                    if filling_type == mt5.ORDER_FILLING_FOK:
                        request["type_filling"] = mt5.ORDER_FILLING_IOC
                    elif filling_type == mt5.ORDER_FILLING_IOC:
                        request["type_filling"] = mt5.ORDER_FILLING_RETURN
                    else:
                        request["type_filling"] = mt5.ORDER_FILLING_FOK
                    
                    print_info(f"Yeni doldurma tipi: {request['type_filling']}")
                    result = mt5.order_send(request)
                    
                    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                        print_success("İkinci deneme başarılı! Emir gönderildi.")
                        print_info(f"Emir ID: {result.order}")
                        return True
                
                return False
            
            # Başarılı emir
            print_success(f"Emir başarıyla gönderildi! Emir ID: {result.order}")
            return True
            
        except Exception as e:
            print_error(f"Emir gönderilirken beklenmeyen hata: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def close_position(self, ticket=None, symbol=None, comment="MT5 Bot Close"):
        """
        Açık pozisyonu kapatır
        
        Parametreler:
        - ticket: Kapatılacak pozisyonun bileti (opsiyonel)
        - symbol: Kapatılacak sembol (opsiyonel, ticket belirtilmezse tüm sembol pozisyonları kapatılır)
        - comment: İşlem için açıklama (opsiyonel)
        
        Başarı durumunda True, başarısızlık durumunda False döndürür
        """
        if not self.connected and not self.connect():
            print_error("MT5 bağlantısı kurulamadı, pozisyon kapatılamadı")
            return False
        
        try:
            # Belirli bir pozisyonu kapat
            if ticket is not None:
                position = mt5.positions_get(ticket=ticket)
                
                if position is None or len(position) == 0:
                    print_error(f"Bilet {ticket} için pozisyon bulunamadı")
                    return False
                
                position = position[0]
                symbol = position.symbol
                
                # İşlem tipi, alım ise satış, satış ise alım olmalı
                action = mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL
                
                # Sembol bilgisini al
                symbol_info = self.symbol_info(symbol)
                if symbol_info is None:
                    return False
                
                # Doldurma politikası ayarları
                filling_modes = symbol_info.filling_mode
                filling_type = None
                
                if filling_modes & mt5.SYMBOL_FILLING_FOK:
                    filling_type = mt5.ORDER_FILLING_FOK
                elif filling_modes & mt5.SYMBOL_FILLING_IOC:
                    filling_type = mt5.ORDER_FILLING_IOC
                else:
                    filling_type = mt5.ORDER_FILLING_RETURN
                
                # Mevcut fiyatı al
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print(f"'{symbol}' için fiyat bilgisi alınamadı, pozisyon kapatılamadı")
                    return False
                
                # Alış/satış fiyatına göre kapatma fiyatını belirle
                price = tick.ask if action == mt5.ORDER_TYPE_BUY else tick.bid
                
                # Kapatma talebini hazırla
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": action,
                    "position": position.ticket,
                    "price": price,
                    "deviation": 10,
                    "magic": 12345,
                    "comment": comment,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": filling_type
                }
                
                # Kapatma emrini gönder
                result = mt5.order_send(request)
                
                if result is None:
                    print(f"Pozisyon kapatılamadı, MT5 hatası: {mt5.last_error()}")
                    return False
                
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"Pozisyon kapatma başarısız oldu, hata kodu: {result.retcode}")
                    return False
                
                print(f"Pozisyon {position.ticket} başarıyla kapatıldı")
                return True
            
            # Belirli bir sembol için tüm pozisyonları kapat
            elif symbol is not None:
                positions = mt5.positions_get(symbol=symbol)
                
                if positions is None or len(positions) == 0:
                    print(f"'{symbol}' için açık pozisyon bulunamadı")
                    return False
                
                success = True
                for position in positions:
                    # Her pozisyonu kapat
                    if not self.close_position(ticket=position.ticket, comment=comment):
                        success = False
                
                return success
            
            # Tüm pozisyonları kapat
            else:
                positions = mt5.positions_get()
                
                if positions is None or len(positions) == 0:
                    print("Açık pozisyon bulunamadı")
                    return False
                
                success = True
                for position in positions:
                    # Her pozisyonu kapat
                    if not self.close_position(ticket=position.ticket, comment=comment):
                        success = False
                
                return success
                
        except Exception as e:
            print(f"Pozisyon kapatılırken hata: {str(e)}")
            return False

    def _get_mt5_timeframe(self, timeframe: str) -> int:
        """
        String olarak verilen zaman dilimini MT5 timeframe'ine çevirir
        
        Parametreler:
        - timeframe: Zaman dilimi (ör. "1h", "4h", "1d")
        
        Dönüş:
        - MT5 timeframe sabiti
        """
        timeframe_map = {
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "1W": mt5.TIMEFRAME_W1,
            "1M": mt5.TIMEFRAME_MN1
        }
        
        if timeframe not in timeframe_map:
            print(f"Geçersiz zaman dilimi: {timeframe}. Desteklenen değerler: {list(timeframe_map.keys())}")
            return None
        
        return timeframe_map[timeframe] 