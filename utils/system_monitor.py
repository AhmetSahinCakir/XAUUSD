import psutil
import logging
import time
from threading import Thread, Event
import gc
from typing import Callable
from config import SYSTEM_CONFIG

logger = logging.getLogger("TradingBot.SystemMonitor")

class SystemMonitor:
    """Sistem kaynaklarını ve bağlantı durumunu izleyen sınıf"""
    
    def __init__(self, mt5_connector, emergency_callback: Callable = None):
        self.mt5 = mt5_connector
        self.emergency_callback = emergency_callback
        self.stop_event = Event()
        self.monitor_thread = None
        
        # Yapılandırma parametreleri
        self.gc_interval = SYSTEM_CONFIG['gc_interval']
        self.max_memory = SYSTEM_CONFIG['max_memory_usage']
        self.reconnect_attempts = SYSTEM_CONFIG['reconnect_attempts']
        self.reconnect_wait = SYSTEM_CONFIG['reconnect_wait']
        self.heartbeat_interval = SYSTEM_CONFIG['heartbeat_interval']
        
        self.last_gc_time = time.time()
        self.last_heartbeat_time = time.time()
    
    def start_monitoring(self):
        """İzleme thread'ini başlat"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("Sistem izleme başlatıldı")
    
    def stop_monitoring(self):
        """İzleme thread'ini durdur"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join()
            logger.info("Sistem izleme durduruldu")
    
    def _monitor_loop(self):
        """Ana izleme döngüsü"""
        while not self.stop_event.is_set():
            try:
                self._check_memory()
                self._check_connection()
                time.sleep(1)  # CPU kullanımını azaltmak için kısa bekleme
            except Exception as e:
                logger.error(f"İzleme döngüsünde hata: {str(e)}")
    
    def _check_memory(self):
        """Bellek kullanımını kontrol et ve gerekirse temizle"""
        current_time = time.time()
        
        # Bellek kullanım oranını kontrol et
        memory_percent = psutil.Process().memory_percent()
        
        # Bellek kullanımı çok yüksekse veya GC zamanı geldiyse
        if memory_percent > self.max_memory * 100 or \
           current_time - self.last_gc_time >= self.gc_interval:
            
            logger.info(f"Bellek temizleniyor (Kullanım: {memory_percent:.1f}%)")
            gc.collect()
            self.last_gc_time = current_time
    
    def _check_connection(self):
        """MT5 bağlantısını kontrol et ve gerekirse yeniden bağlan"""
        current_time = time.time()
        
        # Heartbeat kontrolü
        if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
            if not self.mt5.connected:
                logger.warning("MT5 bağlantısı koptu, yeniden bağlanmaya çalışılıyor...")
                
                for attempt in range(self.reconnect_attempts):
                    if self.mt5.connect():
                        logger.info("MT5 bağlantısı yeniden sağlandı")
                        break
                    
                    if attempt < self.reconnect_attempts - 1:
                        time.sleep(self.reconnect_wait)
                else:
                    logger.error("MT5 yeniden bağlantı başarısız!")
                    if self.emergency_callback:
                        self.emergency_callback()
            
            self.last_heartbeat_time = current_time
    
    def get_system_stats(self):
        """Sistem durumu istatistiklerini döndür"""
        return {
            'memory_usage': psutil.Process().memory_percent(),
            'cpu_usage': psutil.Process().cpu_percent(),
            'connection_status': self.mt5.connected,
            'last_gc_time': time.strftime('%Y-%m-%d %H:%M:%S', 
                                        time.localtime(self.last_gc_time)),
            'last_heartbeat': time.strftime('%Y-%m-%d %H:%M:%S', 
                                          time.localtime(self.last_heartbeat_time))
        } 