"""Offline unit tests for metrics SQL compilation + window parsing.

* ``compile_percentile_sql`` must emit ``WITHIN GROUP (ORDER BY ... ASC)``
  BEFORE its ``FILTER`` (Postgres rejects the reverse order at execution).
* ``parse_range`` resolves 1h/24h/7d/30d, enforces range vs from/to mutual
  exclusion, and rejects naive datetimes.
* ``pick_granularity`` boundary behaviour.

No live database — only the Postgres dialect compiler is used.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.metrics import (
    compile_percentile_sql,
    parse_range,
    pick_granularity,
)

NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=timezone.utc)


# --- compile_percentile_sql -------------------------------------------------


def test_percentile_within_group_precedes_filter():
    sql = compile_percentile_sql()
    within = "WITHIN GROUP (ORDER BY usage_records.latency_ms ASC)"
    assert within in sql
    # Every WITHIN GROUP must come before its FILTER clause.
    first_within = sql.find(within)
    first_filter = sql.find("FILTER", first_within)
    assert first_within != -1
    assert first_filter != -1
    assert first_within < first_filter


def test_percentile_sql_has_three_percentiles():
    sql = compile_percentile_sql()
    assert sql.count("percentile_cont") == 3
    assert "AS p50" in sql
    assert "AS p95" in sql
    assert "AS p99" in sql


# --- parse_range ------------------------------------------------------------


@pytest.mark.parametrize(
    "key,delta",
    [
        ("1h", timedelta(hours=1)),
        ("24h", timedelta(hours=24)),
        ("7d", timedelta(days=7)),
        ("30d", timedelta(days=30)),
    ],
)
def test_parse_range_keywords(key, delta):
    start, end = parse_range(key, None, None, now=NOW)
    assert end == NOW
    assert start == NOW - delta


def test_parse_range_defaults_to_24h():
    start, end = parse_range(None, None, None, now=NOW)
    assert end == NOW
    assert start == NOW - timedelta(hours=24)


def test_parse_range_invalid_keyword_raises():
    with pytest.raises(ValueError):
        parse_range("13h", None, None, now=NOW)


def test_parse_range_explicit_from_to():
    frm = "2026-06-01T00:00:00+00:00"
    to = "2026-06-02T00:00:00+00:00"
    start, end = parse_range(None, frm, to, now=NOW)
    assert start == datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert end == datetime(2026, 6, 2, tzinfo=timezone.utc)


def test_parse_range_from_only_defaults_to_now():
    frm = "2026-06-01T00:00:00+00:00"
    start, end = parse_range(None, frm, None, now=NOW)
    assert end == NOW


def test_parse_range_mutual_exclusion_raises():
    with pytest.raises(ValueError):
        parse_range("1h", "2026-06-01T00:00:00+00:00", None, now=NOW)


def test_parse_range_naive_from_raises():
    with pytest.raises(ValueError):
        parse_range(None, "2026-06-01T00:00:00", None, now=NOW)


def test_parse_range_to_before_from_raises():
    with pytest.raises(ValueError):
        parse_range(
            None,
            "2026-06-02T00:00:00+00:00",
            "2026-06-01T00:00:00+00:00",
            now=NOW,
        )


def test_parse_range_to_without_from_raises():
    with pytest.raises(ValueError):
        parse_range(None, None, "2026-06-02T00:00:00+00:00", now=NOW)


def test_parse_range_naive_now_raises():
    with pytest.raises(ValueError):
        parse_range("1h", None, None, now=datetime(2026, 6, 4, 12, 0, 0))


# --- pick_granularity -------------------------------------------------------


def test_pick_granularity_minute_at_and_below_1h():
    assert pick_granularity(timedelta(hours=1)) == "minute"
    assert pick_granularity(timedelta(minutes=30)) == "minute"


def test_pick_granularity_hour_boundary():
    assert pick_granularity(timedelta(hours=1, seconds=1)) == "hour"
    assert pick_granularity(timedelta(hours=24)) == "hour"


def test_pick_granularity_day_above_24h():
    assert pick_granularity(timedelta(hours=24, seconds=1)) == "day"
    assert pick_granularity(timedelta(days=30)) == "day"
