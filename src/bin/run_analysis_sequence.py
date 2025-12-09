#!/usr/bin/env python3
import subprocess
import time

def run_with_enter(module):
    # Run a python module and send a newline to stdin
    subprocess.run(
        ["python", "-m", module],
        input=b"\n",
        check=True
    )

def main():
    # 1) find_rot_tran (press Enter)
    run_with_enter("src.analysis.find_rot_tran")
    time.sleep(1)

    # 2) performance_analysis (press Enter)
    run_with_enter("src.analysis.performance_analysis")
    time.sleep(1)

    # 3) plotting_csv (no input needed)
    subprocess.run(["python", "-m", "src.analysis.plotting_csv"], check=True)

if __name__ == "__main__":
    main()
