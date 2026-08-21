"""Train the compact slot model on existing shard datasets."""

from __future__ import annotations

from autotrack.dl import train_track_slot as base
from autotrack.dl.compact_slot_model import (
    ModelConfig,
    TrackSlotPredictor,
    save_checkpoint,
    track_slot_detection_metrics,
    track_slot_set_loss,
)

base.ModelConfig = ModelConfig
base.TrackSlotPredictor = TrackSlotPredictor
base.save_checkpoint = save_checkpoint
base.track_slot_detection_metrics = track_slot_detection_metrics
base.track_slot_set_loss = track_slot_set_loss


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
