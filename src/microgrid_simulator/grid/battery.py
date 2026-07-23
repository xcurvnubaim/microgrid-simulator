"""Compatibility shim — moved to ``microgrid_simulator.components.battery``."""

from __future__ import annotations

from microgrid_simulator.components.battery import (
    BatteryLike,
    BatteryModel,
    PymgridBatteryModel,
    create_battery_model,
)

__all__ = ["BatteryLike", "BatteryModel", "PymgridBatteryModel", "create_battery_model"]
