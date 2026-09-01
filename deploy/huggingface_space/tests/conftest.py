from __future__ import annotations

import sys
from pathlib import Path


SPACE_ROOT = Path(__file__).resolve().parents[1]
if str(SPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(SPACE_ROOT))

