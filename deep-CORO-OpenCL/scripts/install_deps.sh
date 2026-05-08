#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U pip
python -m pip install -e .
python -m pip install pyopencl
