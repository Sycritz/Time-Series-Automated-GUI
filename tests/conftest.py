import os
import sys

# Add src/ to path so tests can import modules without package prefixes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
