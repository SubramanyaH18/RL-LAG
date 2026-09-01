"""pytest configuration — add project root to sys.path so tests resolve imports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
