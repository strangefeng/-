from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        str(ROOT / "run_experiment_suite.py"),
        "--manifest",
        str(ROOT / "experiments" / "aircraft_ai_dataset_round9_rtdetr.json"),
    ]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
