"""Time utilities."""

from __future__ import annotations

from datetime import date, datetime
from typing import Union

import pandas as pd

DateLike = Union[str, date, datetime, pd.Timestamp]


def parse_date(d: DateLike) -> date:
    """Parse various date formats to date object."""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise TypeError(f"Cannot parse date from type {type(d)}")


def years_to_expiry(expiry: DateLike, trade_date: DateLike | None = None) -> float:
    """Calculate time to expiry in years."""
    exp_date = parse_date(expiry)
    ref_date = parse_date(trade_date) if trade_date else date.today()
    days = (exp_date - ref_date).days
    return days / 365.0
