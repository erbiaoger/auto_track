"""Compatibility worker entrypoint for the isolated PeakSlot process."""

from autotrack.web_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
