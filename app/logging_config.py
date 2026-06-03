"""Centralised logging setup. No print() anywhere in the codebase.

Supports two formats selected by ``LOG_FORMAT``:
- ``text`` (default): human-readable single-line records.
- ``json``: structured one-object-per-line logs for ingestion.

A ``request_id`` ContextVar is injected into every record (when set by the
request-id middleware), so logs can be correlated to a single request.
"""

import json
import logging
import sys
from contextvars import ContextVar

#: Set per-request by the request-id middleware; empty outside a request.
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class _RequestIdFilter(logging.Filter):
    """Attach the current request id (if any) to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("")
        return True


class _JsonFormatter(logging.Formatter):
    """Render log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", "")
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


#: Marker attribute identifying the handler this module owns.
_OWNED = "_llmgw_handler"


def configure_logging(level: str = "INFO", fmt: str = "text") -> None:
    """Configure root logging with a single stdout handler.

    Idempotent and non-invasive: manages only its own handler, never mutating
    handlers added by other frameworks (e.g. pytest's log capture).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    if fmt == "json":
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    for handler in root.handlers:
        if getattr(handler, _OWNED, False):
            handler.setLevel(level.upper())
            handler.setFormatter(formatter)
            return

    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _OWNED, True)
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
