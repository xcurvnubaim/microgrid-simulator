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
        """Find valid telemetry starting timestamps that fit full episodes within split boundaries."""
        dt_hours = float(self.settings.topology.timestep_hours)
        n_steps = int(round(self.settings.episode.horizon_hours / dt_hours))

        # We inspect available timestamps from digital_twin config if available
        # Default full dataset timestamp range (Dec 2025 - Mar 2026)
        full_range = pd.date_range("2025-12-01 00:00:00", "2026-03-31 23:45:00", freq=f"{int(dt_hours*60)}min")

        valid_starts: list[pd.Timestamp] = []
        episode_timedelta = pd.to_timedelta(self.settings.episode.horizon_hours, unit="h")

        for start in full_range:
            end = start + episode_timedelta

            # Filter by split
            if self.split == "train":
                # Dec 2025 - Jan 2026
                if not (start >= pd.Timestamp("2025-12-01") and end <= pd.Timestamp("2026-02-01")):
                    continue
            elif self.split == "val":
                # Feb 2026 (excluding Feb 4 gap: 2026-02-04 00:00 to 2026-02-05 00:00)
                if not (start >= pd.Timestamp("2026-02-01") and end <= pd.Timestamp("2026-03-01")):
                    continue
                feb4_start = pd.Timestamp("2026-02-04 00:00:00")
                feb4_end = pd.Timestamp("2026-02-05 00:00:00")
                if not (end <= feb4_start or start >= feb4_end):
                    continue
            elif self.split == "test":
                # March 2026
                if not (start >= pd.Timestamp("2026-03-01") and end <= pd.Timestamp("2026-04-01")):
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
        n_steps = int(round(temp_settings.episode.horizon_hours / float(temp_settings.topology.timestep_hours)))
        return load_fixed_telemetry_window(temp_settings, n_steps)
