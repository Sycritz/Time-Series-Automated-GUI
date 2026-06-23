import sys
import os
import runpy

# Ensure src/ is on sys.path so modules can find sibling dependencies
src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

if __name__ == "__main__":
    runpy.run_path(os.path.join(src_dir, "main.py"), run_name="__main__")
