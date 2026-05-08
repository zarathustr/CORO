"""Regenerate manuscript-facing publication figures for the parallel calibration study."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_coro_znn_parallel.publication_figures import regenerate_requested_figures


def main() -> None:
    outputs = regenerate_requested_figures(ROOT)
    print("Regenerated figures:")
    for key, value in outputs.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
