"""Centralised logging setup. No print() anywhere in the codebase."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging with a single stdout handler.

    Idempotent: safe to call more than once (won't stack handlers).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Avoid duplicate handlers if called twice (e.g. tests + app startup).
    if root.handlers:
        for handler in root.handlers:
            handler.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
