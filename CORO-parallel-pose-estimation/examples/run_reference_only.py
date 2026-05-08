
"""Run only the Zhang reference benchmark."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from deep_coro_znn_parallel.zhang_reference import run_zhang_reference_benchmark
def main() -> None:
    output_dir = ROOT / "results" / "reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_zhang_reference_benchmark(output_dir=str(output_dir))
    print(output_dir)
if __name__ == "__main__":
    main()
