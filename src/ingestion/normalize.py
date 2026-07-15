import re
import warnings

import pandas as pd


def _compute_duration(start, end):
    if start is None or end is None:
        return None
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return None
    return int((end_ts - start_ts).total_seconds())
