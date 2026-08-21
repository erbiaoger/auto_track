"""Deprecated compatibility wrapper; implementation lives in PeakSlot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/peak_slot_tracker/src"))
from peak_slot_tracker.peak_slot_model import *  # noqa: F401,F403
