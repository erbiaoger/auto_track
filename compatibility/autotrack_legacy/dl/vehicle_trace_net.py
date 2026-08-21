"""Deprecated compatibility wrapper; implementation lives in SingleVehicleTrace."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/single_vehicle_trace_tracker/src"))
from single_vehicle_trace_tracker.vehicle_trace_net import *  # noqa: F401,F403
from single_vehicle_trace_tracker.vehicle_trace_net import _auto_torch_device  # noqa: F401
