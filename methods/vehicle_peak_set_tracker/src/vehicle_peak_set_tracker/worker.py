"""Compatibility worker entrypoint for the isolated Vehicle Peak-Set process."""

from autotrack.web_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
