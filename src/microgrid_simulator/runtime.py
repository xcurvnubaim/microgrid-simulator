"""Small helpers shared by independently configured runtime services."""

from __future__ import annotations

import hashlib

from microgrid_simulator.config import Settings


def settings_fingerprint(settings: Settings) -> str:
    payload = settings.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
