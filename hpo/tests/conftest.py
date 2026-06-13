"""Test configuration for the HPO src layout."""

from pathlib import Path
import sys


HPO_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
DQN_SRC_DIR = Path(__file__).resolve().parents[2] / "dqn" / "src"
sys.path.insert(0, str(HPO_SRC_DIR))
sys.path.insert(0, str(DQN_SRC_DIR))

