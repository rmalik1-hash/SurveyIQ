import pandas as pd


def question_header(index: int, scale_min: int, scale_max: int) -> str:
    return (
        f"On a scale of {scale_min} to {scale_max}, how much do you agree "
        f"with statement {index + 1}? [Q{index + 1}]"
    )


def attention_check_header(ac_number: int, scale_max: int) -> str:
    return (
        f"For quality control, please select {scale_max} for this item. "
        f"[AC{ac_number}]"
    )


def render_messy_csv(rows: list[dict]) -> pd.DataFrame:
    """Assemble per-respondent flat dicts into a DataFrame, preserving the
    column order of the first row (all rows are expected to share the same
    keys in the same order, as produced by generator.py)."""
    return pd.DataFrame(rows)
