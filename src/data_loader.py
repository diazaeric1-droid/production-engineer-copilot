"""Load and validate well files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WellFile:
    """Normalized representation of a single well's data package."""
    well_id: str
    api_number: str
    field: str
    operator: str
    spud_date: str
    first_prod_date: str
    completion: dict[str, Any]
    artificial_lift: dict[str, Any]
    production_history: list[dict[str, Any]]
    scada_recent: list[dict[str, Any]] = field(default_factory=list)
    dyno_cards: list[dict[str, Any]] = field(default_factory=list)
    esp_readings: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> "WellFile":
        path = Path(path)
        with path.open() as f:
            raw = json.load(f)
        return cls(**raw)

    def summary(self) -> str:
        if self.production_history:
            days_on = max(row.get("day", 0) for row in self.production_history)
        else:
            days_on = 0
        lift = self.artificial_lift.get("type", "unknown")
        n_points = len(self.production_history)
        return (
            f"Well {self.well_id} ({self.api_number}) | {self.field} | "
            f"{lift} | {days_on} days on production ({n_points} data points)"
        )
