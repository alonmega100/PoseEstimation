#!/usr/bin/env python3
import subprocess

def run_with_enter(cmd):
    # Run a command and send one newline to stdin
    subprocess.run(cmd, input=b"\n", check=True)

def main():
    # 1) find_rot_tran.py  (press Enter)
    run_with_enter(["python", "find_rot_tran.py"])

    # 2) performance_analysis.py  (press Enter)
    run_with_enter(["python", "performance_analysis.py"])

    # 3) plotting_csv.py (no input needed)
    subprocess.run(["python", "plotting_csv.py"], check=True)

if __name__ == "__main__":
    main()
