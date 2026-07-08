"""Utility intertie — import/export limits at the point of common coupling.

Limits default to ``None`` (unconstrained), which preserves the historical
behaviour of the simulator; set ``intertie.max_import_mw`` /
``intertie.max_export_mw`` in the config to model a contracted capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

from microgrid_simulator.config import Settings
from microgrid_simulator.core.types import ConstraintViolation


@dataclass
class GridIntertieModel:
    """Point of common coupling with optional import/export limits (MW)."""

    max_import_mw: float | None = None
    max_export_mw: float | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> GridIntertieModel:
        return cls(
            max_import_mw=settings.intertie.max_import_mw,
            max_export_mw=settings.intertie.max_export_mw,
        )

    def clamp(self, p_mw: float) -> tuple[float, ConstraintViolation | None]:
        """Clamp the intertie flow (import > 0, export < 0) to its limits.

        Returns ``(feasible_flow_mw, violation)`` — the violation records how
        much the unconstrained balance wanted to exceed the contract.
        """
        if self.max_import_mw is not None and p_mw > self.max_import_mw:
            return self.max_import_mw, ConstraintViolation(
                kind="import_limit",
                device="grid_intertie",
                magnitude=p_mw - self.max_import_mw,
                limit=self.max_import_mw,
                message=f"import {p_mw:.4f} MW exceeds limit {self.max_import_mw:.4f} MW",
            )
        if self.max_export_mw is not None and -p_mw > self.max_export_mw:
            return -self.max_export_mw, ConstraintViolation(
                kind="export_limit",
                device="grid_intertie",
                magnitude=-p_mw - self.max_export_mw,
                limit=self.max_export_mw,
                message=f"export {-p_mw:.4f} MW exceeds limit {self.max_export_mw:.4f} MW",
            )
        return p_mw, None
