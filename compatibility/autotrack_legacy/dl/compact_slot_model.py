"""Deprecated compatibility wrapper; implementation lives in CompactSlot."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/compact_slot_tracker/src"))
from compact_slot_tracker.compact_slot_model import *  # noqa: F401,F403
