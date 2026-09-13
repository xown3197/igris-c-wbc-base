"""IGRIS-C manufacturer-plant WBC baseline.

Importing this package is CPU-safe. Import :mod:`wbc_base.tasks` only after
Isaac Sim's AppLauncher has created the application.
"""

from .contract import CONTRACT, WBCBaseContract

__all__ = ["CONTRACT", "WBCBaseContract"]

