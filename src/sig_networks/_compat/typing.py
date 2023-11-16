from __future__ import annotations

import sys

if sys.version_info < (3, 8):
    from typing import Literal, Protocol, runtime_checkable
else:
    from typing import Literal, Protocol, runtime_checkable

__all__ = ["Protocol", "runtime_checkable", "Literal"]


def __dir__() -> list[str]:
    return __all__
