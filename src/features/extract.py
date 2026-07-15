import numpy as np
import pandas as pd

AVG_SECONDS_PER_QUESTION = 8


def completion_time_ratio(duration_seconds, num_questions):
    if duration_seconds is None:
        return np.nan
    return duration_seconds / (num_questions * AVG_SECONDS_PER_QUESTION)


def straightlining_score(responses):
    values = list(responses.values())
    n = len(values)
    if n == 0:
        raise ValueError("responses must not be empty")
    modal_count = max(values.count(v) for v in set(values))
    return modal_count / n


def response_variance(responses, scale_min, scale_max):
    values = list(responses.values())
    if len(values) == 0:
        raise ValueError("responses must not be empty")
    if scale_max == scale_min:
        return 0.0
    normalized = [(v - scale_min) / (scale_max - scale_min) for v in values]
    return float(np.std(normalized))


def extreme_response_rate(responses, scale_min, scale_max):
    values = list(responses.values())
    n = len(values)
    if n == 0:
        raise ValueError("responses must not be empty")
    if scale_max == scale_min:
        return 0.0
    extreme = sum(1 for v in values if v == scale_min or v == scale_max)
    return extreme / n
