import psutil
import os
import requests
import time

def test_limits():
    print("--- 🚀 PRE-FLIGHT STRESS TEST ---")
    
    # Check RAM
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / 1024 / 1024
    
    print(f"[MEMORY] Current Base Script Footprint: {ram_mb:.2f} MB")
    if ram_mb > 1024:
        print("[WARNING] Over 1GB limit!")
    else:
        print("[OK] Well within the 1.0GB limit.")
        
    # Check API Proxy connection
    print("\n[NETWORK] Testing OpenEnv Proxy Connection...")
    try:
        response = requests.post("http://127.0.0.1:8000/reset", timeout=5)
        if response.status_code == 200:
            print("[OK] Successfully connected to FastAPI proxy wrapper.")
    except Exception as e:
        print(f"[FAIL] Could not connect to proxy on port 8000. \nMake sure you are running 'uvicorn src.api.server:app --port 8000' in another terminal.\nError: {e}")
    
    print("\n[INFO] Stress test complete. Ready for evaluation!")

if __name__ == "__main__":
    test_limits()
