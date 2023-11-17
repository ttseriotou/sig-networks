from __future__ import annotations

import sig_networks as m
from sig_networks._compat.typing import Protocol, runtime_checkable


def test_version():
    assert m.__version__


@runtime_checkable
class HasQuack(Protocol):
    def quack() -> str:
        ...


class Duck:
    def quack() -> str:
        return "quack"


def test_has_typing():
    assert isinstance(Duck(), HasQuack)
