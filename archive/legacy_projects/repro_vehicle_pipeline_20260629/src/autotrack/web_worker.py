"""Entry-point shim for the root method-worker protocol in this environment."""
from pathlib import Path
import runpy

ROOT_WORKER = Path(__file__).resolve().parents[3] / "autotrack" / "web_worker.py"
runpy.run_path(str(ROOT_WORKER), run_name="__main__")
