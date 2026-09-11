"""Point the whole test session at a scratch data directory.

This has to happen here rather than in an individual test module. Pytest
imports every test module during collection, and the first one to import
`swarkavach.config` fixes `DATA_DIR` for the process. Collection is
alphabetical, so `test_antispoof.py` gets there first and it does not set
`SWARKAVACH_DATA`, which means a module that sets the variable at its own
import time is setting it too late and silently operates on the real `data/`.

That is not a hypothetical. It wrote stub archives into `data/raw`, and
`fetch-data` skips any archive that already exists, so a test run before the
download step left the downloader convinced it had already fetched GramVaani
when what it had was 100 bytes of zeros.

conftest.py is imported before any test module, so setting it here wins.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

_SESSION_DATA = Path(tempfile.mkdtemp(prefix="swarkavach_tests_"))
os.environ["SWARKAVACH_DATA"] = str(_SESSION_DATA)


def pytest_sessionstart(session):
    import swarkavach.config as _cfg

    real = ROOT / "data"
    if _cfg.DATA_DIR.resolve() == real.resolve():
        raise RuntimeError(
            "tests are pointed at the real data directory. Something imported "
            "swarkavach.config before conftest ran, and a test run would "
            "overwrite corpora and models."
        )


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_SESSION_DATA, ignore_errors=True)
