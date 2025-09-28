# Test configuration utilities.
# Ensures the repository root is on sys.path so that 'utils', 'ocr', etc. can be imported
# when running pytest without installing the package.
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from utils.llm_usage_tracking import log_global_token_usage



ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Log total LLM token consumption after the full pytest suite finishes."""
    del session, exitstatus
    log_global_token_usage(prefix="pytest llm token usage")
