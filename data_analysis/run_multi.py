import subprocess
import sys
import time
from datetime import datetime

def run_script(path,n):
    for i in range(n):
        print("\n" + "="*50)
        print(f"Starting run {i+1} of {n}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50 + "\n")
        
        try:
            subprocess.run([sys.executable,path],check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in run {i+1}: {e}")

        if i<n-1:
            time.sleep(5)


if __name__== "main":
    path = sys.argv[1]
    run_script(path, 20)