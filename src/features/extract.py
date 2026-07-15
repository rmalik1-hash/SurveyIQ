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


def attention_check_pass_rate(attention_checks):
    given_keys = [k for k in attention_checks if k.endswith("_given")]
    total = 0
    correct = 0
    for given_key in given_keys:
        prefix = given_key[: -len("_given")]
        correct_key = prefix + "_correct"
        if correct_key in attention_checks:
            total += 1
            if attention_checks[given_key] == attention_checks[correct_key]:
                correct += 1
    if total == 0:
        return 1.0
    return correct / total


def _mirror(value, scale_min, scale_max):
    return scale_min + scale_max - value


def contradiction_score(responses, scale_min, scale_max, pairs, tolerance=1):
    if not pairs:
        return 0.0
    contradicting = 0
    for a_key, b_key in pairs:
        if a_key not in responses or b_key not in responses:
            raise ValueError(
                f"contradiction pair ({a_key}, {b_key}) references a missing response"
            )
        gap = abs(responses[b_key] - _mirror(responses[a_key], scale_min, scale_max))
        if gap > tolerance:
            contradicting += 1
    return contradicting / len(pairs)
