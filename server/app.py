from __future__ import annotations

import os
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

try:
    from . import main as _main
except ImportError:
    import importlib
    _main = importlib.import_module("main")

app = _main.app


def main(host: str = "0.0.0.0", port: int | None = None):
    import uvicorn

    if port is None:
        port = int(os.getenv("API_PORT", "7860"))

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
