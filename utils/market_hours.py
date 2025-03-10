import logging
from datetime import datetime, timedelta
import pytz
import json
from pathlib import Path
from typing import Optional, Dict, List
from config import MARKET_HOURS, MARKET_CHECK_INTERVALS

logger = logging.getLogger("TradingBot.MarketHours")

class MarketHours:
    """Piyasa saatlerini ve durumunu yöneten sınıf"""
    
    def __init__(self):
        self.market_hours = MARKET_HOURS
        self.holidays = self._load_holidays()
        self.last_session_check = datetime.now()
        self.last_holiday_check = datetime.now()
        self.last_maintenance_check = datetime.now()
    
    def _load_holidays(self) -> Dict:
        """Tatil günlerini yükle"""
        try:
            holiday_file = Path('data/market_holidays.json')
            if holiday_file.exists():
                with open(holiday_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Tatil günleri yüklenirken hata: {str(e)}")
            return {}
    
    def _convert_to_gmt(self, time_str: str, timezone_str: str) -> datetime:
        """Yerel saati GMT'ye çevir"""
        try:
            local_tz = pytz.timezone(timezone_str)
            gmt_tz = pytz.timezone('GMT')
            
            # Bugünün tarihini al
            now = datetime.now(local_tz)
            
            # Saat ve dakikayı ayır
            hour, minute = map(int, time_str.split(':'))
            
            # Yerel tarih/saat oluştur
            local_dt = local_tz.localize(datetime(
                now.year, now.month, now.day, hour, minute))
            
            # GMT'ye çevir
            return local_dt.astimezone(gmt_tz)
            
        except Exception as e:
            logger.error(f"Saat dönüşümünde hata: {str(e)}")
            return None
    
    def is_market_open(self) -> bool:
        """
        Piyasanın açık olup olmadığını kontrol et
        
        Kontroller:
        1. Hafta sonu kontrolü
        2. Tatil günü kontrolü
        3. Bakım/rollover periyodu kontrolü
        4. Normal piyasa saatleri kontrolü
        """
        now = datetime.now(pytz.timezone('GMT'))
        
        # Hafta sonu kontrolü
        if self._is_weekend():
            logger.debug("Piyasa kapalı: Hafta sonu")
            return False
        
        # Tatil günü kontrolü
        if self._is_holiday():
            logger.debug("Piyasa kapalı: Tatil günü")
            return False
        
        # Bakım/rollover kontrolü
        if self._is_maintenance_time():
            logger.debug("Piyasa kapalı: Bakım/rollover periyodu")
            return False
        
        # Normal piyasa saatleri kontrolü
        return self._is_within_trading_hours()
    
    def _is_weekend(self) -> bool:
        """Hafta sonu kontrolü"""
        now = datetime.now(pytz.timezone('GMT'))
        weekend_start = self._convert_to_gmt(
            self.market_hours['weekend_breaks']['start']['time'],
            self.market_hours['weekend_breaks']['start']['timezone']
        )
        weekend_end = self._convert_to_gmt(
            self.market_hours['weekend_breaks']['end']['time'],
            self.market_hours['weekend_breaks']['end']['timezone']
        )
        
        # Cuma kapanıştan Pazar açılışa kadar kapalı
        if now.weekday() == 4 and now.time() >= weekend_start.time():  # Cuma
            return True
        if now.weekday() == 5:  # Cumartesi
            return True
        if now.weekday() == 6 and now.time() < weekend_end.time():  # Pazar
            return True
            
        return False
    
    def _is_holiday(self) -> bool:
        """Tatil günü kontrolü"""
        now = datetime.now()
        today_str = now.strftime('%Y-%m-%d')
        
        current_year = str(now.year)
        if current_year in self.holidays:
            for holiday in self.holidays[current_year]:
                if holiday['date'] == today_str:
                    if 'All' in holiday['markets']:
                        return True
                    # Erken kapanış kontrolü
                    if 'note' in holiday and 'Early close' in holiday['note']:
                        early_close_time = self._convert_to_gmt(
                            self.market_hours['early_close']['US']['time'],
                            self.market_hours['early_close']['US']['timezone']
                        )
                        if now.time() >= early_close_time.time():
                            return True
        return False
    
    def _is_maintenance_time(self) -> bool:
        """Bakım/rollover periyodu kontrolü"""
        now = datetime.now(pytz.timezone('GMT'))
        
        for break_period in self.market_hours['trading_breaks']['daily']:
            start_time = self._convert_to_gmt(
                break_period['start']['time'],
                break_period['start']['timezone']
            )
            end_time = self._convert_to_gmt(
                break_period['end']['time'],
                break_period['end']['timezone']
            )
            
            if start_time.time() <= now.time() <= end_time.time():
                return True
        
        return False
    
    def _is_within_trading_hours(self) -> bool:
        """Normal piyasa saatleri kontrolü"""
        now = datetime.now(pytz.timezone('GMT'))
        
        # En az bir seans açık olmalı
        for session, hours in self.market_hours['sessions'].items():
            session_start = self._convert_to_gmt(
                hours['open']['time'],
                hours['open']['timezone']
            )
            session_end = self._convert_to_gmt(
                hours['close']['time'],
                hours['close']['timezone']
            )
            
            # Seans saatleri içinde mi?
            if session_start.time() <= now.time() <= session_end.time():
                return True
        
        return False
    
    def get_current_sessions(self) -> List[str]:
        """Şu anda aktif olan seansları döndür"""
        active_sessions = []
        now = datetime.now(pytz.timezone('GMT'))
        
        for session, hours in self.market_hours['sessions'].items():
            session_start = self._convert_to_gmt(
                hours['open']['time'],
                hours['open']['timezone']
            )
            session_end = self._convert_to_gmt(
                hours['close']['time'],
                hours['close']['timezone']
            )
            
            if session_start.time() <= now.time() <= session_end.time():
                active_sessions.append(session)
        
        return active_sessions
    
    def get_next_session(self) -> Optional[Dict]:
        """Bir sonraki seansın bilgilerini döndür"""
        now = datetime.now(pytz.timezone('GMT'))
        next_session = None
        min_time_diff = timedelta(days=1)
        
        for session, hours in self.market_hours['sessions'].items():
            session_start = self._convert_to_gmt(
                hours['open']['time'],
                hours['open']['timezone']
            )
            
            # Eğer seans başlangıcı şu andan sonraysa
            if session_start.time() > now.time():
                time_diff = datetime.combine(now.date(), session_start.time()) - \
                           datetime.combine(now.date(), now.time())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    next_session = {
                        'name': session,
                        'start_time': session_start.time(),
                        'time_until': time_diff
                    }
        
        return next_session
    
    def should_check_market_status(self) -> bool:
        """Piyasa durumu kontrolü yapılmalı mı?"""
        now = datetime.now()
        
        # Seans kontrolü
        if (now - self.last_session_check).total_seconds() >= \
           MARKET_CHECK_INTERVALS['session_check']:
            self.last_session_check = now
            return True
        
        # Tatil günü kontrolü
        if (now - self.last_holiday_check).total_seconds() >= \
           MARKET_CHECK_INTERVALS['holiday_check']:
            self.last_holiday_check = now
            return True
        
        # Bakım/rollover kontrolü
        if (now - self.last_maintenance_check).total_seconds() >= \
           MARKET_CHECK_INTERVALS['maintenance_check']:
            self.last_maintenance_check = now
            return True
        
        return False 