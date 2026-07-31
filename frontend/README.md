# SurveyIQ dashboard

The React frontend for SurveyIQ. Upload a survey CSV, tag its columns, and see
which responses are trustworthy — with a plain-English reason for every flag.

## Running it

The dashboard needs the API running first.

**1. Start the API** (from the `SurveyIQ` project root):

```bash
python -m uvicorn src.api.main:app --port 8000
```

The API trains its model on synthetic data at startup, so give it a second or
two before the first request.

**2. Start the dashboard** (from this `frontend` directory):

```bash
npm install     # first time only
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173).

If your API runs somewhere other than `http://127.0.0.1:8000`, set
`VITE_API_BASE`:

```bash
VITE_API_BASE=http://localhost:9000 npm run dev
```

## Tests

```bash
npm test
```

## How it fits together

1. **Upload** — the file goes to `POST /columns`; the browser also parses it
   locally, which is what makes the cleaned-CSV download possible without
   re-uploading.
2. **Map** — you tag each column, set the scale, give each attention check its
   correct answer, and optionally mark reverse-coded question pairs. This is
   sent to `POST /analyze`.
3. **Results** — quality gauge, flagged respondents with their reasons, the most
   problematic questions, and a download (marked or cleaned).

Responses are never stored: the API analyzes the upload in memory and discards it.
If you give a survey a name on the mapping step, that run's aggregate totals
(date, name, response count, quality score, and the average answer per question) are
recorded so the dashboard can show
quality trends across analyses. No answers, respondent IDs or demographics are
ever written to disk. History lives in `data/history.json` and can be cleared from
the results page.

## Layout

```
src/
  App.jsx                       step machine: upload -> map -> results
  lib/api.js                    calls the API
  lib/mapping.js                builds and validates the mapping object
  lib/cleanedCsv.js             builds the marked/cleaned download
  components/UploadStep.jsx
  components/MappingStep.jsx
  components/ResultsDashboard.jsx
  components/ChartBoundary.jsx  keeps a broken chart from killing the page
tests/                          vitest unit and component tests
```
