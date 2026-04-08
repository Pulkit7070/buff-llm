"""OpenEnv-compatible app entry point — re-exports the FastAPI app from webapp.py."""
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from webapp import app  # noqa: F401, E402


def main():
    """Entry point for OpenEnv server."""
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
