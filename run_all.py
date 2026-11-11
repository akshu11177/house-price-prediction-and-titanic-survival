"""Run both assignment models. Place the Kaggle CSV files in data/ first."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
scripts = [
    ROOT / "src" / "house_price_model.py",
    ROOT / "src" / "titanic_model.py",
]

for script in scripts:
    print(f"\n{'=' * 70}\nRunning {script.name}\n{'=' * 70}")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

print("\nAll models completed successfully. See the outputs/ folder.")
