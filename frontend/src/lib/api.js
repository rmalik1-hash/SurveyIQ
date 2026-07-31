const BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function errorMessage(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    return `Request failed (${res.status})`;
  }
}

export async function fetchColumns(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/columns`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorMessage(res));
  const body = await res.json();
  return body.columns;
}

/** Ask the API for a synthetic survey so the dashboard can be tried without real data. */
export async function generateSampleSurvey({
  nRespondents = 150,
  nQuestions = 15,
  contaminationRate = 0.25,
} = {}) {
  const form = new FormData();
  form.append("n_respondents", String(nRespondents));
  form.append("n_questions", String(nQuestions));
  form.append("contamination_rate", String(contaminationRate));
  const res = await fetch(`${BASE}/generate`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorMessage(res));
  const blob = await res.blob();
  return new File([blob], "synthetic_survey.csv", { type: "text/csv" });
}

export async function analyzeSurvey(file, mapping, surveyLabel = "") {
  const form = new FormData();
  form.append("file", file);
  form.append("mapping", JSON.stringify(mapping));
  // Only sent when the user opted into trend tracking by naming the survey.
  if (surveyLabel) form.append("survey_label", surveyLabel);
  const res = await fetch(`${BASE}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorMessage(res));
  return await res.json();
}

export async function fetchHistory(surveyLabel = "") {
  const query = surveyLabel ? `?survey_label=${encodeURIComponent(surveyLabel)}` : "";
  const res = await fetch(`${BASE}/history${query}`);
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()).runs;
}

export async function clearHistory() {
  const res = await fetch(`${BASE}/history`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()).removed;
}
