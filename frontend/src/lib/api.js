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

export async function analyzeSurvey(file, mapping) {
  const form = new FormData();
  form.append("file", file);
  form.append("mapping", JSON.stringify(mapping));
  const res = await fetch(`${BASE}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorMessage(res));
  return await res.json();
}
