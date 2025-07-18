#!/usr/bin/env python3
import os
import subprocess
import sys

DOWNLOAD_DIR = os.path.dirname(os.path.abspath(__file__))

success = []
failure = []

for fname in os.listdir(DOWNLOAD_DIR):
    if fname.endswith('.py') and fname != os.path.basename(__file__):
        fpath = os.path.join(DOWNLOAD_DIR, fname)
        print(f"Running {fname} ...", flush=True)
        try:
            result = subprocess.run([sys.executable, fpath], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"SUCCESS: {fname}")
                success.append(fname)
            else:
                print(f"FAIL: {fname}\n{result.stderr}")
                failure.append(fname)
        except Exception as e:
            print(f"ERROR running {fname}: {e}")
            failure.append(fname)

print("\nSummary:")
print(f"  Success: {success}")
print(f"  Failure: {failure}")
if failure:
    sys.exit(1)
