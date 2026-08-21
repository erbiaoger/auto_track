"""Deprecated compatibility wrapper; implementation lives in Proposal/Trace."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/vehicle_proposal_trace_pipeline/src"))
from vehicle_proposal_trace_pipeline.vehicle_proposal_net import *  # noqa: F401,F403
