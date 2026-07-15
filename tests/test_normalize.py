import pandas as pd
import pytest
from src.ingestion.normalize import _compute_duration


def test_compute_duration_normal():
    assert _compute_duration("2024-03-01T08:00:00", "2024-03-01T08:02:00") == 120


def test_compute_duration_missing_returns_none():
    assert _compute_duration(None, "2024-03-01T08:02:00") is None
    assert _compute_duration("2024-03-01T08:00:00", None) is None


def test_compute_duration_unparseable_returns_none():
    assert _compute_duration("not a date", "2024-03-01T08:02:00") is None
