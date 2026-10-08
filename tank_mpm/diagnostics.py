from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConstraintDiagnostics:
    step: int
    side: str
    max_pin_gap: float
    rms_pin_gap: float
    max_center_spacing_error: float
    max_orientation_error: float
    mean_orientation_error: float


def write_constraint_diagnostics(path: Path | str, rows: list[ConstraintDiagnostics]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(ConstraintDiagnostics.__dataclass_fields__.keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fields})
