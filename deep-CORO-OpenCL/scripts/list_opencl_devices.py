from __future__ import annotations
import json
from deep_coro_opencl.opencl_operator import list_opencl_devices

if __name__ == "__main__":
    print(json.dumps(list_opencl_devices(), indent=2))
