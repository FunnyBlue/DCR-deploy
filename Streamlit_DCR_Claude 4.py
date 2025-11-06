import os
import signal
import subprocess
import sys
import time
import requests
import webbrowser
import platform
from pathlib import Path

# ====== Kill anything already on :8501 (cross-platform best-effort) ======
def kill_on_port(port=8501):
    try:
        system = platform.system().lower()
        if "windows" in system:
            out = subprocess.check_output(
                ["cmd", "/c", f"netstat -ano | findstr :{port}"], text=True, stderr=subprocess.DEVNULL
            )
            pids = set()
            for line in out.splitlines():
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pids.add(parts[-1])
            for pid in pids:
                try:
                    subprocess.run(["taskkill", "/PID", pid, "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
        else:
            pids = subprocess.check_output(
                ["bash", "-lc", f"lsof -t -i:{port}"], text=True
            ).strip().split()
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass

# ====== Health check function ======
def wait_health(url="http://127.0.0.1:8501/health", timeout=40):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

# ====== Main execution ======
if __name__ == "__main__":
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ Error: app.py not found in the current directory!")
        print("Please ensure app.py is in the same folder as run_local.py")
        sys.exit(1)
    
    print("🔍 Checking for existing processes on port 8501...")
    kill_on_port(8501)
    
    # Start Streamlit
    cmd = [
        "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1",
        "--server.headless", "true"
    ]
    
    print("🚀 Starting Streamlit application...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Wait for health check
    print("⏳ Waiting for Streamlit to start on http://127.0.0.1:8501 ...")
    if not wait_health():
        print("❌ Streamlit failed to pass health check. Logs:")
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                print(line, end="")
        except Exception:
            pass
        sys.exit(1)
    
    print("✅ Streamlit is up at: http://127.0.0.1:8501")
    
    # Open browser
    try:
        webbrowser.open("http://127.0.0.1:8501")
        print("🌐 Browser opened automatically")
    except Exception:
        print("⚠️  Could not open browser automatically. Please visit: http://127.0.0.1:8501")
    
    print("\n" + "="*60)
    print("--- Streaming Streamlit logs (Ctrl+C to stop) ---")
    print("="*60 + "\n")
    
    # Stream logs
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.2)
                continue
            print(line, end="")
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("👋 Shutting down Streamlit...")
        print("="*60)
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("✅ Streamlit stopped successfully")