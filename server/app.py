"""OpenEnv-compatible app entry point — re-exports the FastAPI app from webapp.py."""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp import app  # noqa: F401, E402
