"""
Stage 4: Run remaining ablations A1, A2, A4, A5, A6, A7, A8, A9 sequentially.
Execute AFTER A3 finishes to avoid CPU contention.
"""

from __future__ import annotations
import sys, subprocess
from pathlib import Path

ROOT = Path("d:/Documents/Autism_Project")
PYTHON = r"C:\Users\user\AppData\Local\Programs\Python\Python312\python.exe"

REMAINING = [
    "a1",
    "a2",
    "a4",
    "a5",
    "a6",
    "a7",
    "a8",
    "a9"
]

if __name__ == "__main__":
    for exp in REMAINING:
        print(f"\n{'='*70}\nLaunching experiment: {exp.upper()}\n{'='*70}")
        result = subprocess.run(
            [PYTHON, "-u",
             str(ROOT / "research_stage4" / "experiments" / "run_stage4_ablations.py"),
             "--exp", exp, "--seed", "42"],
            cwd=str(ROOT)
        )
        if result.returncode != 0:
            print(f"ERROR: Experiment {exp} exited with code {result.returncode}")
        else:
            print(f"Experiment {exp} completed successfully.")
    print("\nAll remaining Stage 4 ablations finished.")
