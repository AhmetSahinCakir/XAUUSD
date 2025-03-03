from mt5_connector import MT5Connector
import pandas as pd

def check_data_sizes():
    """
    Checks and compares data sizes for different timeframes
    """
    mt5 = MT5Connector()
    
    # Number of candles to test
    candle_counts = {
        "1m": [1000, 5000, 10000],
        "5m": [1000, 2000, 5000],
        "15m": [500, 1000, 2000]
    }
    
    results = {}
    
    print("\n=== Data Size Check ===")
    print("--------------------------------")
    
    for timeframe in candle_counts.keys():
        print(f"\n{timeframe} Timeframe Results:")
        print("--------------------------------")
        results[timeframe] = {}
        
        for n_candles in candle_counts[timeframe]:
            data = mt5.get_historical_data(timeframe, n_candles=n_candles)
            
            if data is not None:
                actual_size = len(data)
                coverage = (actual_size / n_candles) * 100
                memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024  # in MB
                
                results[timeframe][n_candles] = {
                    "requested_candles": n_candles,
                    "received_candles": actual_size,
                    "coverage_ratio": coverage,
                    "memory_usage": memory_usage
                }
                
                print(f"\nRequested Candles: {n_candles}")
                print(f"Received Candles: {actual_size}")
                print(f"Coverage Ratio: {coverage:.2f}%")
                print(f"Memory Usage: {memory_usage:.2f} MB")
            else:
                print(f"\nRequested Candles: {n_candles}")
                print("Failed to get data!")
    
    return results

if __name__ == "__main__":
    check_data_sizes() 