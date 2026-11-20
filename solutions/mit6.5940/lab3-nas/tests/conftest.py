import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "code"))
sys.path.insert(0, str(LAB.parents[2]))
