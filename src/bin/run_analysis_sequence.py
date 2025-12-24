#!/usr/bin/env python3
import subprocess
import sys
import os
from src.utils.tools import find_latest_csv

def run_step(module, csv_path):
    print(f"\n>>> Running {module} with {csv_path}...")
    subprocess.run(
        ["python", "-m", module, "--csv", csv_path],
        check=True
    )

def main():
    # 1. Determine the file ONCE
    csv_dir = "data/CSV"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        print("Finding latest CSV automatically...")
        csv_path = find_latest_csv(csv_dir)
        if not csv_path:
            print("No CSV found.")
            return

    print(f"Targeting CSV: {csv_path}")

    # 2. Run sequence explicitly passing the file
    # This prevents the sub-scripts from stopping for input
    try:
        run_step("src.analysis.find_rot_tran", csv_path)
        run_step("src.analysis.performance_analysis", csv_path)
        run_step("src.analysis.plotting_csv", csv_path)
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed at step: {e.cmd}")

if __name__ == "__main__":
    main()