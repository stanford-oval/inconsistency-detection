"""High-level inconsistency detection interfaces."""

from .agent import InconsistencyAgent
from .models import (
    InconsistencyReport,
)


__all__ = [
    "InconsistencyAgent",
    "InconsistencyReport",
]
