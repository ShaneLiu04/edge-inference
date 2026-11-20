import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "code"))
sys.path.insert(0, str(PROJ.parents[2]))
