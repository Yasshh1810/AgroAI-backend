import subprocess
import sys
import os
import webbrowser
import time

print("=" * 45)
print("  AgroAI - Plant Leaf Disease Detection")
print("=" * 45)

# Install required packages if missing
PACKAGES = ["fastapi", "uvicorn", "python-multipart", "pillow"]

print("\nChecking dependencies...")
for pkg in PACKAGES:
    try:
        __import__(pkg.replace("-", "_"))
        print(f"  OK  {pkg}")
    except ImportError:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        print(f"  OK  {pkg}")

print("\nStarting AgroAI server...")
print("Open your browser at: http://localhost:8000")
print("\nPress CTRL+C to stop the server.\n")

# Auto open browser after 2 seconds
def open_browser():
    time.sleep(2)
    webbrowser.open("http://localhost:8000")

import threading
threading.Thread(target=open_browser, daemon=True).start()

# Start server
os.system(f"{sys.executable} -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload")
