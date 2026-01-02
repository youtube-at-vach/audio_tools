from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on sys.path so imports like `from src...` work
# when pytest is invoked via the `pytest` entrypoint script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
