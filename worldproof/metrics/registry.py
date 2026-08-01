"""Metric registry: name -> Metric class, with duplicate-name protection.

Metrics register themselves in the global :data:`REGISTRY` (via the
:func:`register` decorator) so the runner and CLI can look them up by name.
Names must be unique — a metric's cited numbers are keyed on its name+version,
so two different metrics under one name would make results ambiguous.
"""

from __future__ import annotations

from worldproof.metrics.base import Metric

__all__ = ["MetricRegistry", "REGISTRY", "register"]


class MetricRegistry:
    """A name → Metric-class map. Re-registering the same class is idempotent."""

    def __init__(self) -> None:
        self._by_name: dict[str, type[Metric]] = {}

    def register(self, cls: type[Metric]) -> type[Metric]:
        if not (isinstance(cls, type) and issubclass(cls, Metric)):
            raise TypeError(f"register expects a Metric subclass, got {cls!r}")
        existing = self._by_name.get(cls.name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"a different metric is already registered under name "
                f"{cls.name!r} ({existing.__name__}); metric names must be unique"
            )
        self._by_name[cls.name] = cls
        return cls

    def get(self, name: str) -> type[Metric]:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"no metric registered under {name!r}; registered: {self.names()}"
            ) from None

    def create(self, name: str, **kwargs: object) -> Metric:
        return self.get(name)(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    def classes(self) -> list[type[Metric]]:
        return [self._by_name[name] for name in self.names()]

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self._by_name)


REGISTRY = MetricRegistry()


def register(cls: type[Metric]) -> type[Metric]:
    """Class decorator: register ``cls`` in the global :data:`REGISTRY`."""
    return REGISTRY.register(cls)
