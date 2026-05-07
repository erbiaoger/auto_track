# autotrack.simulation

Synthetic DAS data generation helpers.

- `simulate_vehicle_sac_torch.py`: accelerated generator for SAC files,
  `tracks.json`, `vehicles.csv`, and preview outputs.

TrackSlotNet training usually uses `autotrack.dl.generate_track_slot_dataset`
instead, because it writes tensor shards directly and skips SAC I/O.
