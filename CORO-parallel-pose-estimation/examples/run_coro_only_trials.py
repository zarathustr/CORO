
"""Run the CORO-only (>100 trial) calibration benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_coro_znn_parallel.coro_only_trials import benchmark_coro_only_trials


def main() -> None:
    output_dir = ROOT / "results" / "coro_only_mc120"
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_coro_only_trials(num_trials=120, seed=29, output_dir=str(output_dir))
    print(f"Wrote CORO-only 120-trial benchmark to: {output_dir}")


if __name__ == "__main__":
    main()
