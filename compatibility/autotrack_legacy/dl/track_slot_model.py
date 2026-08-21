"""Deprecated compatibility wrapper; implementation lives in TrackSlot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/track_slot_tracker/src"))
from track_slot_tracker.track_slot_model import *  # noqa: F401,F403
