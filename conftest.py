import sys
import os

# Add src/ to path so src modules can import each other by bare name
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
