"""Episode sampling and dataset split logic for RL training and evaluation.

Train: Dec 2025 – Jan 2026
Val: Feb 2026 (excluding 2026-02-04 gap)
Test: March 2026
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from microgrid_simulator.config import Settings
from microgrid_simulator.digital_twin.data_ingestion import load_measurements
from microgrid_simulator.digital_twin.replay import TelemetryWindow, load_fixed_telemetry_window

LOGGER = logging.getLogger(__name__)

SplitName = Literal["train", "val", "test", "all"]


class RandomEpisodeSampler:
    """Samples valid 72-hour (or custom length) episode telemetry windows by split."""

    def __init__(
        self,
        settings: Settings,
        split: SplitName = "train",
        seed: int | None = None,
    ) -> None:
        self.settings = settings
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.valid_start_timestamps = self._build_valid_starts()

    def _build_valid_starts(self) -> list[pd.Timestamp]:
        """Find starts that fit complete episodes within split boundaries."""
        dt_hours = float(self.settings.topology.timestep_hours)
        n_steps = int(round(self.settings.episode.horizon_hours / dt_hours))

        configured = self.settings.digital_twin.measurements
        if all(name in configured for name in ("load", "pv")):
            measurements = load_measurements(self.settings.digital_twin)
            common = measurements["load"].index.intersection(measurements["pv"].index)
            full_range = pd.DatetimeIndex(common).sort_values()
        else:
            # Keep the default synthetic/demand-trace configuration usable for
            # existing smoke tests that do not configure telemetry files.
            full_range = pd.date_range(
                "2025-12-01 00:00:00",
                "2026-03-31 23:45:00",
                freq=f"{int(dt_hours * 60)}min",
            )

        valid_starts: list[pd.Timestamp] = []
        # Split boundaries describe the half-open episode interval, while the
        # replay loader separately includes one preceding context sample.
        episode_timedelta = pd.to_timedelta(n_steps * dt_hours, unit="h")

        for start in full_range:
            end = start + episode_timedelta
            context = start - pd.to_timedelta(dt_hours, unit="h")
            expected = pd.date_range(context, end, freq=pd.to_timedelta(dt_hours, unit="h"))
            if configured and not expected.isin(full_range).all():
                continue

            # Filter by split
            if self.split == "train":
                # Dec 2025 - Jan 2026
                if not (
                    start >= pd.Timestamp("2025-12-02 00:15:00")
                    and end <= pd.Timestamp("2026-01-30")
                ):
                    continue
            elif self.split == "val":
                # Manifest-backed February validation interval.
                if not (
                    start >= pd.Timestamp("2026-02-06")
                    and end <= pd.Timestamp("2026-02-25")
                ):
                    continue
            elif self.split == "test" and not (
                start >= pd.Timestamp("2026-03-01")
                and end <= pd.Timestamp("2026-04-01")
            ):
                    continue

            valid_starts.append(start)

        if not valid_starts:
            LOGGER.warning("No valid episode start timestamps found for split %r", self.split)
        return valid_starts

    def sample_window(self) -> TelemetryWindow | None:
        """Sample a TelemetryWindow from the valid timestamps."""
        if not self.valid_start_timestamps:
            return None
        start_ts = self.rng.choice(self.valid_start_timestamps)
        # Create temporary settings with this start timestamp
        cfg_dump = self.settings.model_dump()
        cfg_dump["episode"]["telemetry_start"] = start_ts.strftime("%Y-%m-%d %H:%M:%S")
        temp_settings = Settings(**cfg_dump)
        n_steps = int(
            round(
                temp_settings.episode.horizon_hours
                / float(temp_settings.topology.timestep_hours)
            )
        )
        return load_fixed_telemetry_window(temp_settings, n_steps)
