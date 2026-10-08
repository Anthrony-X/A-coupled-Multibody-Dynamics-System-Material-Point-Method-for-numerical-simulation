from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tank_mpm.chrono_loader import require_chrono_core  # noqa: E402


def main() -> int:
    chrono = require_chrono_core()
    system = chrono.ChSystemSMC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, -9.81, 0.0))

    body = chrono.ChBody()
    body.SetMass(10.0)
    body.SetInertiaXX(chrono.ChVector3d(1.0, 1.0, 1.0))
    body.SetPos(chrono.ChVector3d(0.0, 1.0, 0.0))
    system.AddBody(body)

    dt = 1.0e-3
    for _ in range(10):
        system.DoStepDynamics(dt)

    pos = body.GetPos()
    print(
        "[Done] Chrono core smoke: "
        f"bodies={system.GetNumBodies()}, contacts={system.GetNumContacts()}, "
        f"t={system.GetChTime():.6f}s, y={pos.y:.9f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
