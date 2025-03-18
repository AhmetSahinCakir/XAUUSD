import psutil
import logging
import time
from threading import Thread, Event
import gc
from typing import Callable
from config.config import SYSTEM_CONFIG
from utils.logger import print_info, print_warning

logger = logging.getLogger("TradingBot.SystemMonitor")

class SystemMonitor:
    """Sistem kaynaklarını ve bağlantı durumunu izleyen sınıf"""
    
    def __init__(self, mt5_connector=None, emergency_callback: Callable = None):
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
        
        self.last_check = time.time()
        self.check_interval = 60  # 60 saniye
    
    def start_monitoring(self):
        """Start monitoring thread"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print_info(
                "Sistem izleme başlatıldı",
                "System monitoring started"
            )
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_event.set()
            self.monitor_thread.join()
            print_info(
                "Sistem izleme durduruldu",
                "System monitoring stopped"
            )
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                # Check memory usage
                memory = psutil.virtual_memory()
                if memory.available < 2 * 1024 * 1024 * 1024:  # Less than 2GB
                    print_warning(
                        f"Düşük bellek: {memory.available / (1024*1024*1024):.1f}GB kullanılabilir. Performans düşük olabilir.",
                        f"Low memory: {memory.available / (1024*1024*1024):.1f}GB available. Performance may be degraded."
                    )
                    
                # Check MT5 connection
                if self.mt5 and not self.mt5.connected:
                    print_warning(
                        "MT5 bağlantısı koptu! Yeniden bağlanmaya çalışılıyor...",
                        "MT5 connection lost! Attempting to reconnect..."
                    )
                    if not self._attempt_reconnect():
                        print_error(
                            "MT5 yeniden bağlantı başarısız!",
                            "MT5 reconnection failed!"
                        )
                        if self.emergency_callback:
                            self.emergency_callback()
                            
                # Periodic garbage collection
                if time.time() - self.last_gc_time > self.gc_interval:
                    gc.collect()
                    self.last_gc_time = time.time()
                    
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                print_error(
                    f"İzleme döngüsü hatası: {str(e)}",
                    f"Monitoring loop error: {str(e)}"
                )
                time.sleep(5)  # Wait before retrying
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to MT5"""
        for attempt in range(self.reconnect_attempts):
            try:
                print_info(
                    f"MT5 yeniden bağlantı denemesi {attempt + 1}/{self.reconnect_attempts}",
                    f"MT5 reconnection attempt {attempt + 1}/{self.reconnect_attempts}"
                )
                if self.mt5.connect():
                    print_success(
                        "MT5 yeniden bağlantı başarılı!",
                        "MT5 reconnection successful!"
                    )
                    return True
                time.sleep(self.reconnect_wait)
            except Exception as e:
                print_error(
                    f"Yeniden bağlantı hatası: {str(e)}",
                    f"Reconnection error: {str(e)}"
                )
        return False
    
    def check(self):
        """Sistem durumunu kontrol et"""
        current_time = time.time()
        if current_time - self.last_check >= self.check_interval:
            self.last_check = current_time
            self._check_memory()
            self._check_cpu()
            self._check_disk()
    
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
        """MT5 bağlantısını kontrol et"""
        try:
            if self.mt5 is None:
                return True
                
            if not self.mt5.connected:
                logger.warning("MT5 bağlantısı koptu, yeniden bağlanmaya çalışılıyor...")
                
                for attempt in range(self.reconnect_attempts):
                    logger.info(f"Yeniden bağlanma denemesi {attempt+1}/{self.reconnect_attempts}")
                    
                    if self.mt5.connect():
                        logger.info("MT5 bağlantısı yeniden kuruldu")
                        return True
                    
                    time.sleep(self.reconnect_wait)
                
                logger.error(f"MT5 yeniden bağlantı başarısız ({self.reconnect_attempts} deneme)")
                
                if self.emergency_callback:
                    logger.warning("Acil durum prosedürü başlatılıyor!")
                    self.emergency_callback()
                return False
            
            # Heartbeat kontrolü
            current_time = time.time()
            if current_time - self.last_heartbeat_time > self.heartbeat_interval:
                self.last_heartbeat_time = current_time
                account_info = self.mt5.get_account_info()
                if account_info is None:
                    logger.warning("MT5 heartbeat başarısız, yeniden bağlanmaya çalışılıyor...")
                    return self.mt5.connect()
            
            return True
                
        except Exception as e:
            logger.error(f"Bağlantı kontrolü hatası: {str(e)}")
            return False
    
    def _check_cpu(self):
        """CPU kullanımını kontrol et"""
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            logger.warning("⚠️ Yüksek CPU kullanımı: %{:.1f}".format(cpu_percent))
            
    def _check_disk(self):
        """Disk kullanımını kontrol et"""
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            logger.warning("⚠️ Yüksek disk kullanımı: %{:.1f}".format(disk.percent))
    
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
        
    def check_status(self):
        """Sistemin genel durumunu kontrol et
        
        Dönüş:
        - True: Sistem normal çalışıyor
        - False: Kritik bir sorun var
        """
        try:
            # MT5 bağlantısını kontrol et
            if not self.mt5.connected:
                logger.error("MT5 bağlantısı koptu!")
                return False
                
            # Bellek kullanımını kontrol et
            memory_percent = psutil.Process().memory_percent()
            if memory_percent > 95:  # %95'ten fazla bellek kullanımı kritik
                logger.error(f"Kritik bellek kullanımı: %{memory_percent:.1f}")
                return False
                
            # CPU kullanımını kontrol et
            cpu_percent = psutil.cpu_percent(interval=0.5)
            if cpu_percent > 95:  # %95'ten fazla CPU kullanımı kritik
                logger.error(f"Kritik CPU kullanımı: %{cpu_percent:.1f}")
                return False
                
            # Disk kullanımını kontrol et
            disk = psutil.disk_usage('/')
            if disk.percent > 95:  # %95'ten fazla disk kullanımı kritik
                logger.error(f"Kritik disk kullanımı: %{disk.percent:.1f}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Sistem durumu kontrolü sırasında hata: {str(e)}")
            return False 