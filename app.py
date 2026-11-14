"""SafeVision Edge — Root Application Entry Point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to sys.path so modules resolve cleanly
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import the FastAPI application
from api import app  # noqa: E402

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SAFEVISION_HOST", "0.0.0.0")
    port = int(os.getenv("SAFEVISION_PORT", "8000"))
    print(f"Starting SafeVision Edge on http://{host}:{port}/ ...")
    uvicorn.run("app:app", host=host, port=port, reload=False)
