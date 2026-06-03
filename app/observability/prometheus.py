"""A tiny, dependency-free Prometheus metrics registry + text exposition.

We deliberately hand-roll a minimal Counter/Histogram instead of pulling in
``prometheus-client`` (keeps the runtime dependency footprint at zero new
packages). Thread-safe via a single lock; cardinality is bounded by the small
set of (provider, model, status) label combinations a gateway sees.

Exposes one entry point for the request pipeline, :func:`observe_request`, and
one for the ``/metrics`` endpoint, :func:`render_prometheus`.
"""

import threading
from collections.abc import Sequence

_LOCK = threading.Lock()

# Latency histogram buckets, in milliseconds (upper bounds).
_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000,
)


def _fmt(value: float) -> str:
    """Render a float the Prometheus way (integers without trailing .0)."""
    if value == int(value):
        return str(int(value))
    return repr(value)


def _labels_str(labelnames: Sequence[str], labelvalues: Sequence[str]) -> str:
    if not labelnames:
        return ""
    parts = []
    for name, raw in zip(labelnames, labelvalues):
        escaped = (
            str(raw)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        parts.append(f'{name}="{escaped}"')
    return "{" + ",".join(parts) + "}"


class _Counter:
    def __init__(self, name: str, help_text: str, labelnames: Sequence[str]):
        self.name = name
        self.help_text = help_text
        self.labelnames = tuple(labelnames)
        self._values: dict[tuple[str, ...], float] = {}

    def inc(self, amount: float = 1.0, *, labels: Sequence[str] = ()) -> None:
        key = tuple(labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help_text}", f"# TYPE {self.name} counter"]
        for key, value in sorted(self._values.items()):
            lines.append(f"{self.name}{_labels_str(self.labelnames, key)} {_fmt(value)}")
        return lines


class _Histogram:
    def __init__(
        self,
        name: str,
        help_text: str,
        labelnames: Sequence[str],
        buckets: Sequence[float],
    ):
        self.name = name
        self.help_text = help_text
        self.labelnames = tuple(labelnames)
        self.buckets = tuple(buckets)
        self._bucket_counts: dict[tuple[str, ...], list[int]] = {}
        self._sums: dict[tuple[str, ...], float] = {}
        self._counts: dict[tuple[str, ...], int] = {}

    def observe(self, value: float, *, labels: Sequence[str] = ()) -> None:
        key = tuple(labels)
        counts = self._bucket_counts.get(key)
        if counts is None:
            counts = [0] * len(self.buckets)
            self._bucket_counts[key] = counts
            self._sums[key] = 0.0
            self._counts[key] = 0
        for i, upper in enumerate(self.buckets):
            if value <= upper:
                counts[i] += 1
        self._sums[key] += value
        self._counts[key] += 1

    def render(self) -> list[str]:
        lines = [
            f"# HELP {self.name} {self.help_text}",
            f"# TYPE {self.name} histogram",
        ]
        for key in sorted(self._bucket_counts):
            # counts[i] is already cumulative ("observations <= buckets[i]").
            counts = self._bucket_counts[key]
            for upper, count in zip(self.buckets, counts):
                le_labels = (*self.labelnames, "le")
                le_values = (*key, _fmt(upper))
                lines.append(
                    f"{self.name}_bucket{_labels_str(le_labels, le_values)} {count}"
                )
            inf_labels = (*self.labelnames, "le")
            inf_values = (*key, "+Inf")
            total = self._counts[key]
            lines.append(
                f"{self.name}_bucket{_labels_str(inf_labels, inf_values)} {total}"
            )
            lines.append(
                f"{self.name}_sum{_labels_str(self.labelnames, key)} {_fmt(self._sums[key])}"
            )
            lines.append(
                f"{self.name}_count{_labels_str(self.labelnames, key)} {total}"
            )
        return lines


# --- Metric instances ---
_REQUESTS = _Counter(
    "llmgw_requests_total",
    "Total chat requests routed through the gateway.",
    ("provider", "model", "status"),
)
_TOKENS = _Counter(
    "llmgw_tokens_total",
    "Total tokens processed.",
    ("provider", "model", "kind"),
)
_COST = _Counter(
    "llmgw_cost_usd_total",
    "Total computed cost in USD.",
    ("provider", "model"),
)
_LATENCY = _Histogram(
    "llmgw_request_latency_ms",
    "Provider request latency in milliseconds.",
    ("provider", "model"),
    _LATENCY_BUCKETS_MS,
)

_ALL = (_REQUESTS, _TOKENS, _COST, _LATENCY)


def observe_request(
    *,
    provider: str,
    model: str,
    status: str,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Record one completed (or failed) request into the metrics registry."""
    with _LOCK:
        _REQUESTS.inc(labels=(provider, model, status))
        _LATENCY.observe(latency_ms, labels=(provider, model))
        if prompt_tokens:
            _TOKENS.inc(prompt_tokens, labels=(provider, model, "prompt"))
        if completion_tokens:
            _TOKENS.inc(completion_tokens, labels=(provider, model, "completion"))
        if cost_usd:
            _COST.inc(cost_usd, labels=(provider, model))


def render_prometheus() -> str:
    """Render all metrics in the Prometheus text exposition format."""
    with _LOCK:
        lines: list[str] = []
        for metric in _ALL:
            lines.extend(metric.render())
        return "\n".join(lines) + "\n"


def reset() -> None:
    """Clear all metrics (used in tests)."""
    with _LOCK:
        for metric in _ALL:
            metric.__init__(  # type: ignore[misc]
                metric.name,
                metric.help_text,
                metric.labelnames,
                *([metric.buckets] if isinstance(metric, _Histogram) else []),
            )
