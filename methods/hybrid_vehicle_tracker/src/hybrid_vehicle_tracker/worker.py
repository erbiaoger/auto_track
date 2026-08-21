"""Hybrid worker facade used by the shared replay coordinator."""

from vehicle_replay_web.worker import main


if __name__ == "__main__":
    raise SystemExit(main())
