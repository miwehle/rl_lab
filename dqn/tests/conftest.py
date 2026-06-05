"""Test configuration for the DQN src layout."""

from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(TESTS_DIR))
