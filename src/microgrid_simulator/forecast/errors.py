"""Forecast exception hierarchy.

``ForecastSourceError`` exists so a response or cache belonging to a different
scenario is rejected loudly instead of being silently consumed.
"""

from __future__ import annotations


class ForecastError(RuntimeError):
    """The forecast service failed or returned an invalid payload."""


class ForecastSourceError(ForecastError):
    """A response belongs to a different scenario and must not be consumed."""

    def __init__(self, expected_source: str, actual_source: str):
        super().__init__(f"forecast source is incompatible with {expected_source}")
        self.expected_source = expected_source
        self.actual_source = actual_source


class ForecastCacheError(ForecastError):
    """An on-disk forecast cache or its manifest failed strict validation."""
