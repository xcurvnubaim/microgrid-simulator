"""Client boundary for the separately deployed Chronos PV/demand forecaster.

The simulator intentionally talks to ``microgrid-forecaster`` over HTTP rather
than importing its heavy foundation-model stack. One request is made at episode
reset. Forecast values are observations for controllers/agents and dashboard
comparison only; the physical backend continues to use its own trajectories.

The package is split by responsibility:

* ``errors``   exception hierarchy shared by the HTTP client and the cache
* ``context``  request-side MW history serialized to the API's kW contract
* ``snapshot`` validated, timestamp-aligned PV/demand forecast container
* ``client``   standard-library HTTP client (+ cache-aware variant)
* ``cache``    on-disk JSONL cache, manifest, and strict leakage-free loading
"""

from microgrid_simulator.forecast.cache import ForecastCache, ForecastCacheManifest
from microgrid_simulator.forecast.client import (
    CachedForecastClient,
    ForecastClient,
    StrictCachedForecastClient,
)
from microgrid_simulator.forecast.context import ForecastContext
from microgrid_simulator.forecast.errors import (
    ForecastCacheError,
    ForecastError,
    ForecastSourceError,
)
from microgrid_simulator.forecast.snapshot import ForecastSnapshot

__all__ = [
    "CachedForecastClient",
    "ForecastCache",
    "ForecastCacheError",
    "ForecastCacheManifest",
    "ForecastClient",
    "ForecastContext",
    "ForecastError",
    "ForecastSnapshot",
    "ForecastSourceError",
    "StrictCachedForecastClient",
]
