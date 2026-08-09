"""Public X8 clean-room audit contract interface.

Independent analysis and comparison are intentionally added in the audit task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path("output/results/evidence_audit_x8_c1_r1/clean_room_contract.json")


def load_contract(repo_root: Path) -> dict[str, Any]:
    """Load the frozen X8 contract for a future audit invocation."""
    with (repo_root / CONTRACT_PATH).open(encoding="utf-8") as handle:
        return json.load(handle)
