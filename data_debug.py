import os
import sys
import pandas as pd
import time
import logging
import torch
import numpy as np
from datetime import datetime

# Sistem yolunu ayarla
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MT5 bağlantısı ve veri işleme
from utils.mt5_connector import MT5Connector
from utils.data_processor import DataProcessor

# Test fonksiyonu
def test_trading_bot():
    print("\n==== Test Başlatılıyor ====")
    
    # MT5'e bağlan
    mt5 = MT5Connector()
    if not mt5.connect():
        print("MT5 bağlantısı başarısız!")
        return
    
    print(f"MT5 bağlantısı başarılı. Hesap: {mt5.get_account_info().login}")
    
    # Veri işleme
    data_processor = DataProcessor()
    
    # Farklı zaman dilimlerinden veri al
    timeframes = ["1m", "5m", "15m"]
    for timeframe in timeframes:
        try:
            # Verileri al
            print(f"\nFetching data for {timeframe}...")
            candles = 1000 if timeframe == "15m" else (2000 if timeframe == "5m" else 5000)
            data = mt5.get_historical_data("XAUUSD", timeframe, num_candles=candles)
            
            if data is None:
                print(f"Veri alınamadı - {timeframe}")
                continue
            
            print(f"Received {len(data)} candles for {timeframe}")
            
            # Eksik sütunları kontrol et
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                print(f"Eksik sütunlar: {missing_columns}")
                continue
            
            # Test 1: Teknik göstergeleri hesapla
            try:
                print(f"Technical indicators hesaplanıyor...")
                if 'tick_volume' not in data.columns:
                    print("'tick_volume' eksik, default değerle ekleniyor")
                    data['tick_volume'] = 1
                
                data_with_indicators = data_processor.add_technical_indicators(data)
                print(f"Göstergeler hesaplandı. Sütunlar: {data_with_indicators.columns.tolist()}")
            except Exception as e:
                print(f"Teknik gösterge hatası: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # Test 2: RL state oluştur
            try:
                print(f"RL state oluşturuluyor...")
                state = data_processor.prepare_rl_state(data.iloc[-1])
                print(f"RL state oluşturuldu. Şekil: {state.shape}")
            except Exception as e:
                print(f"RL state hatası: {str(e)}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Genel hata - {timeframe}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Bağlantıyı kapat
    mt5.disconnect()
    print("\n==== Test Tamamlandı ====")

if __name__ == "__main__":
    # Test'i çalıştır
    test_trading_bot() 