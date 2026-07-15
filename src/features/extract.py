import numpy as np
import pandas as pd

AVG_SECONDS_PER_QUESTION = 8


def completion_time_ratio(duration_seconds, num_questions):
    if duration_seconds is None:
        return np.nan
    return duration_seconds / (num_questions * AVG_SECONDS_PER_QUESTION)
