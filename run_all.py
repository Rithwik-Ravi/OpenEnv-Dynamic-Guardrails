import os
import sys
import time
import psutil
import subprocess
import signal

PORTS_TO_CHECK = [8000, 8001]
processes = []

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.net_connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"[CLEANUP] Found process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}. Terminating...")
                    proc.kill()
                    proc.wait(timeout=3)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def cleanup_ports():
    print("[CLEANUP] Checking for zombied ports...")
    for port in PORTS_TO_CHECK:
        kill_process_on_port(port)

def start_background_process(args_list, name):
    print(f"[START] Launching {name} in background...")
    proc = subprocess.Popen(args_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc

def cleanup_servers(signum=None, frame=None):
    print("\n[SHUTDOWN] Terminating background servers...")
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    cleanup_ports()
    print("[SHUTDOWN] Complete.")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, cleanup_servers)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, cleanup_servers)

    cleanup_ports()
    
    python_exe = sys.executable
    
    # Launch Proxy using sys.executable -m uvicorn to ensure it binds within the venv perfectly
    p1 = start_background_process([python_exe, "-m", "uvicorn", "src.api.server:app", "--port", "8000"], "Core API Server")
    processes.append(p1)
    
    # Launch UI
    p2 = start_background_process([python_exe, "-m", "uvicorn", "src.ui.dashboard:app", "--port", "8001"], "Telemetry UI Server")
    processes.append(p2)
    
    print("[WAIT] Allowing servers to initialize (2 seconds)...")
    time.sleep(2)
    
    print("\n[EVALUATION] Starting Headless Evaluator...\n")
    try:
        subprocess.run([python_exe, "src/inference/evaluate.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Evaluator failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        pass
    
    print("\n[EVALUATION] Finished.")
    print("[READY] Servers are still running in background. View UI at http://127.0.0.1:8001")
    print("[READY] Press Ctrl+C to shutdown completely.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup_servers()

if __name__ == "__main__":
    main()
