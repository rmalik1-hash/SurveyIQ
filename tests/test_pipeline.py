import json
import pandas as pd
from data.synthetic.generator import generate_survey_csv
from src.ingestion.normalize import apply_mapping
from src.features.extract import extract_features


def _build_mapping(columns):
    col_roles = {}
    ac_answers = {}
    for c in columns:
        if "[Q" in c:
            col_roles[c] = "question"
        elif "[AC" in c:
            col_roles[c] = "attention_check"
            ac_answers[c] = 5
        elif c == "Response ID":
            col_roles[c] = "respondent_id"
        elif c == "Start Time":
            col_roles[c] = "start_time"
        elif c == "Timestamp":
            col_roles[c] = "end_time"
        elif c == "Email Address":
            col_roles[c] = "ignore"
        else:
            col_roles[c] = "demographic"
    return {"columns": col_roles, "scale": [1, 5], "attention_check_answers": ac_answers}


def test_full_pipeline_separates_archetypes(tmp_path):
    out = tmp_path / "s.csv"
    generate_survey_csv(
        n_respondents=40, n_questions=10, scale=(1, 5),
        contamination_rate=0.4, seed=5, output_path=str(out),
    )
    df = pd.read_csv(out)
    labels = pd.read_csv(tmp_path / "s_labels.csv")
    pairs_idx = json.load(open(tmp_path / "s_pairs.json"))["pairs"]

    respondents = apply_mapping(df, _build_mapping(list(df.columns)))
    pair_keys = [(f"q{a + 1}", f"q{b + 1}") for a, b in pairs_idx]
    features = extract_features(respondents, contradiction_pairs=pair_keys)

    merged = features.merge(labels, on="respondent_id")

    straight = merged[merged["archetype"] == "straightliner"]
    assert len(straight) > 0
    assert (straight["straightlining_score"] == 1.0).all()

    speed = merged[merged["archetype"] == "speeder"]
    assert len(speed) > 0
    assert (speed["completion_time_ratio"] < 0.5).all()

    reliable = merged[merged["archetype"] == "reliable"]
    assert (reliable["contradiction_score"] == 0.0).all()
    assert (reliable["attention_check_pass_rate"] == 1.0).all()
