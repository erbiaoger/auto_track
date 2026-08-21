"""Hybrid deep-learning and physics-guided DAS vehicle tracking."""

from .config import TrackerConfig, load_tracker_config
from .tracker import HybridVehicleTracker
from .types import TrackBatch, VehicleObservation, VehicleTrack

__all__ = [
    "HybridVehicleTracker",
    "TrackBatch",
    "TrackerConfig",
    "VehicleObservation",
    "VehicleTrack",
    "load_tracker_config",
]

__version__ = "0.1.0"
