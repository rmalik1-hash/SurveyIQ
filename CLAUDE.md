# SurveyIQ — Claude Code Briefing

## What this project is

SurveyIQ is a web-based tool that takes raw survey CSV exports from schools and tells administrators which responses to trust. It detects careless responding — straightlining, random answering, logical contradictions, speeding — and outputs a reliability score per respondent with a human-readable explanation for every flag.

The end user is a school administrator (e.g. at IMSA) who uploads a survey CSV and gets back a dashboard showing overall data quality, which respondents were flagged and why, and which questions generated the most inconsistencies.

This is a SAIL AI Innovation Fellowship project. We are three Grade 11 students at IMSA (Illinois Mathematics and Science Academy). The team is Rayan Malik, Kaustubh Bukkapatnam, and Pranav Gadde. We work collaboratively across all parts of the codebase — there are no fixed lanes.

---

## Tech stack

- **Language:** Python 3.11
- **ML:** scikit-learn (decision trees specifically — chosen because they produce human-readable rules administrators can understand)
- **Data processing:** pandas, numpy
- **Backend API:** FastAPI
- **Frontend:** React (simple dashboard — not complex)
- **Visualization:** Plotly
- **Package management:** pip with requirements.txt

No heavy frameworks. No unnecessary dependencies. Keep it simple and auditable.

---

## Folder structure

```
surveyiq/
  data/
    synthetic/        ← generated fake survey CSVs for development
    real/             ← real IMSA survey data (not available yet, do not create)
    samples/          ← small sample files for testing
  src/
    ingestion/        ← CSV upload, column mapping, normalization
    features/         ← feature extraction from normalized survey data
    models/           ← decision tree training, scoring, rule extraction
    api/              ← FastAPI backend
  frontend/           ← React dashboard
  notebooks/          ← exploratory analysis, not production code
  tests/              ← unit tests
  requirements.txt
  README.md
```

---

## Data flow (end to end)

```
Raw CSV upload
    ↓
Column mapper (user tags which column is what)
    ↓
Normalizer → standard internal schema
    ↓
Feature extractor → feature DataFrame (one row per respondent)
    ↓
Decision tree classifier → reliability score + flag reason per respondent
    ↓
Dashboard output
```

Each stage is a clean function with a defined input/output contract. They do not bleed into each other.

---

## Internal schema (what everything normalizes to)

After ingestion, every survey is represented as a list of respondent dicts:

```python
{
    "respondent_id": str,
    "start_time": datetime,        # may be None if not provided
    "end_time": datetime,          # may be None if not provided
    "duration_seconds": int,       # computed from start/end; None if times missing
    "responses": {
        "q1": int,                 # Likert value, normalized to 0-1 scale internally
        "q2": int,
        # ...
    },
    "attention_checks": {
        "ac1_given": int,          # what respondent answered
        "ac1_correct": int,        # what the correct answer is
    },
    "demographics": {},            # optional, passed through untouched
    "scale_min": int,              # e.g. 1 for a 1-5 scale
    "scale_max": int,              # e.g. 5 for a 1-5 scale
}
```

---

## Features to extract (src/features/)

These are the behavioral signals that feed into the classifier. Each is computed per respondent from the normalized schema.

| Feature | Description | How to compute |
|---|---|---|
| `completion_time_ratio` | Actual time vs expected reading time | `duration_seconds / (num_questions * AVG_SECONDS_PER_QUESTION)` where AVG_SECONDS_PER_QUESTION = 8 |
| `straightlining_score` | Fraction of questions answered with identical value | `count(modal_response) / num_questions` |
| `response_variance` | Spread of Likert answers (normalized 0-1) | `std(normalized_responses)` |
| `contradiction_score` | Fraction of logically paired questions that contradict | Requires contradiction pairs to be defined at upload time; 0 if none defined |
| `attention_check_pass_rate` | Fraction of attention checks answered correctly | `correct_checks / total_checks`; 1.0 if no attention checks |
| `extreme_response_rate` | Fraction of responses at scale endpoints (min or max) | `count(scale_min or scale_max) / num_questions` |

All features are floats in range [0, 1] where meaningful. Missing data (e.g. no timestamps) results in `None` for that feature — the model must handle sparse features gracefully.

---

## ML model (src/models/)

- **Algorithm:** Decision tree classifier (scikit-learn `DecisionTreeClassifier`)
- **Target:** Binary label — reliable (0) or flagged (1)
- **Training data source:** Synthetic data initially; real IMSA data later after calibration
- **Why decision trees:** Every flag must come with a human-readable rule. e.g. "Completion time under 45 seconds AND straightlining above 80% → flagged." Random forests or neural nets are not acceptable for v1 because they can't explain themselves to a school administrator.
- **Output per respondent:** reliability score (probability of being reliable, float 0-1) + the specific rule path that triggered the flag

The model exposes two functions:
```python
train(feature_df, labels) -> model
predict(model, feature_df) -> DataFrame with columns: respondent_id, reliability_score, flag_reason
```

---

## Synthetic data generator (data/synthetic/)

This is the first thing to build. We do not have real survey data yet. The generator must produce realistic fake survey CSVs that include:

- A mix of reliable respondents (varied answers, reasonable time, pass attention checks)
- Straightliners (same answer to everything)
- Speeders (completed in implausibly short time)
- Random responders (high variance, fail attention checks)
- Contradictors (logically inconsistent on paired questions)

The generator should be configurable — number of respondents, number of questions, scale type, contamination rate (what fraction are careless). Output is a raw CSV in a format a school might actually export (messy column names, etc.) — not pre-cleaned.

```python
# Target interface
generate_survey_csv(
    n_respondents=200,
    n_questions=20,
    scale=(1, 5),
    contamination_rate=0.25,   # 25% careless respondents
    seed=42,
    output_path="data/synthetic/survey_001.csv"
)
```

---

## Column mapping (src/ingestion/)

Schools export CSVs from Google Forms, Qualtrics, SurveyMonkey — all different formats. We do not force schools to reformat their data. Instead:

1. Accept any CSV
2. Show the user their column names
3. Ask them to tag each column: respondent ID / start timestamp / end timestamp / survey question (Likert) / attention check / demographic / ignore

The mapper stores the mapping so the same school never has to do it twice. After mapping, the normalizer converts everything to the internal schema above.

```python
# Target interface
apply_mapping(raw_df, mapping_dict) -> normalized_respondent_list
```

Likert scale mismatch (1-5 vs 1-7 vs 1-10): normalize all responses to [0, 1] during ingestion using `(value - scale_min) / (scale_max - scale_min)`.

---

## What to build first (priority order)

1. **Synthetic data generator** — everything else depends on having data
2. **Feature extractor** — core of the product; takes normalized data, returns feature DataFrame
3. **Normalizer** — converts raw CSV + mapping to internal schema
4. **Decision tree trainer + predictor** — takes feature DataFrame, returns scores + explanations
5. **FastAPI backend** — thin wrapper around the above
6. **React dashboard** — last; only after the pipeline works end-to-end

Do not build the frontend until the backend pipeline works. Do not build the API until feature extraction and scoring work.

---

## Testing approach

- **Phase 1 (now):** Develop and calibrate entirely on synthetic data. The generator produces known ground truth (we know exactly who is careless), so we can measure accuracy properly.
- **Phase 2 (later):** Switch to real IMSA survey data once the pipeline is validated on synthetic. Real data requires administrative approval — it is not available yet. Do not create placeholder files for it.

All feature extraction functions should have unit tests with small hand-crafted inputs where the expected output is obvious. e.g. a respondent who answered 3 to every question should have straightlining_score = 1.0.

---

## Hard constraints

- **No identifiable student data** ever touches the system. The normalizer should strip or ignore any column that looks like a name, email, or ID number that isn't the anonymous respondent token.
- **Every flag must be explainable** in plain English. If you can't generate a human-readable reason for a flag, the approach is wrong.
- **Decision trees only for v1.** No ensemble methods, no neural nets, no black boxes.
- **Keep scope tight.** We have one summer. Survey design feedback, NLP on open-ended responses, and multi-platform auto-ingestion are all out of scope for v1.

---

## What "done" looks like for v1

A school administrator can:
1. Upload a CSV from their survey tool of choice
2. Map their columns once (takes 2 minutes)
3. See a dashboard showing: overall data quality score, list of flagged respondents with scores and plain-English explanations, which questions generated the most inconsistencies
4. Download a cleaned dataset with the unreliable responses removed or marked

That is the complete v1. Nothing else.
