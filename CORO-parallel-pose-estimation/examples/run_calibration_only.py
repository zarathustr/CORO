
"""Run only the corrected mixed calibration benchmark."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from deep_coro_znn_parallel.calibration import benchmark_calibration_suite
def main() -> None:
    output_dir = ROOT / "results" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_calibration_suite(num_trials=80, seed=7, output_dir=str(output_dir))
    print(output_dir)
if __name__ == "__main__":
    main()
