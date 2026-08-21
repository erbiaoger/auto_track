"""Deprecated compatibility wrapper; implementation lives in QueryMask."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/query_mask_tracker/src"))
from query_mask_tracker.query_mask_instance_model import *  # noqa: F401,F403
