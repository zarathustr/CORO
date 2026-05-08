
"""Run all corrected benchmarks and export CSV/PNG artifacts."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from deep_coro_znn_parallel.calibration import benchmark_calibration_suite
from deep_coro_znn_parallel.zhang_reference import run_zhang_reference_benchmark
def main() -> None:
    reference_dir = ROOT / "results" / "reference"
    calibration_dir = ROOT / "results" / "calibration"
    reference_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    run_zhang_reference_benchmark(output_dir=str(reference_dir))
    benchmark_calibration_suite(num_trials=80, seed=7, output_dir=str(calibration_dir))
    print("Finished. Results written to:")
    print(reference_dir)
    print(calibration_dir)
if __name__ == "__main__":
    main()
