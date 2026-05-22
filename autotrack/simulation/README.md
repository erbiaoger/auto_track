# autotrack.simulation

Synthetic DAS data generation helpers.

- `simulate_vehicle_sac_torch.py`: accelerated generator for SAC files,
  `tracks.json`, `vehicles.csv`, and preview outputs.
- `simulate_vehicle_sac_sparse_artifacts.py`: SAC generator for the case where
  the background should not contain continuous small vibration noise. It
  produces isolated Gaussian artifact peaks together with many dead channels and
  missing channel-time blocks.
- `profile_real_npy_background.py`: profile sparse real-background statistics
  from a `.npy` file and write `profile.json`, `realism_profile.json`, and
  `report.md`. `realism_profile.json` contains generator defaults, a weighted
  window catalog, and unlabeled vehicle proxy statistics for profile-driven
  simulation.
- `simulate_vehicle_sac_from_real_npy.py`: sample a real `.npy` background
  window and overlay simulated vehicles onto it, producing GUI-loadable SAC
  outputs with far smaller train/real domain gap.

TrackSlotNet training usually uses `autotrack.dl.generate_track_slot_dataset`
instead, because it writes tensor shards directly and skips SAC I/O.
