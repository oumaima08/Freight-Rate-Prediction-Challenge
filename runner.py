"""Run the whole pipeline from the project root (works on Windows, macOS and Linux)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(script_name, *args):
    """Run <script_name> (path relative to the project root) with the current Python interpreter, from the project root."""
    print(f"Running {script_name}...")
    command = [sys.executable, str(ROOT / script_name), *args]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        print(f"Error executing {script_name}")
        sys.exit(1)


def main():
    run_script("src/compare_models.py")
    run_script("src/grid_search.py")
    run_script("src/train_predict.py")


if __name__ == "__main__":
    main()