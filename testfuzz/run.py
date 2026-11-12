#!/usr/bin/env python3 -I
"""Isolated-mode launcher for the generative verification runner.

Python's isolated mode (``-I``) ignores PYTHONPATH and the current working
directory, so the repository root is inserted explicitly. This keeps the
harness safe against planted modules (for example a hostile ``json.py`` in
whatever directory the operator happens to stand in) while still importing
the in-tree ``testfuzz`` package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from testfuzz.run_fuzz import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
