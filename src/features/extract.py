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
