import pandas as pd
from data.synthetic.template import (
    question_header,
    attention_check_header,
    render_messy_csv,
)


def test_question_header_contains_question_tag_and_scale():
    header = question_header(index=2, scale_min=1, scale_max=5)
    assert "[Q3]" in header
    assert "1" in header and "5" in header


def test_attention_check_header_contains_tag_and_target():
    header = attention_check_header(ac_number=1, scale_max=5)
    assert "[AC1]" in header
    assert "5" in header


def test_render_messy_csv_preserves_rows_and_column_order():
    rows = [
        {"Response ID": "R0001", "Start Time": "2024-01-01T08:00:00", "Timestamp": "2024-01-01T08:05:00"},
        {"Response ID": "R0002", "Start Time": "2024-01-01T08:10:00", "Timestamp": "2024-01-01T08:15:00"},
    ]
    df = render_messy_csv(rows)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Response ID", "Start Time", "Timestamp"]
    assert df.iloc[0]["Response ID"] == "R0001"
    assert len(df) == 2
