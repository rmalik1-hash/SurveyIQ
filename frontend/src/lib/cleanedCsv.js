export const FLAG_THRESHOLD = 0.5;

/**
 * Join the API's per-respondent results back onto the original uploaded rows.
 * mode "marked" keeps every row and adds the score/reason columns;
 * mode "cleaned" additionally drops the flagged rows.
 */
export function buildCleanedRows(rows, respondents, idColumn, mode) {
  const byId = new Map(respondents.map((r) => [String(r.respondent_id), r]));
  const out = [];
  for (const row of rows) {
    const result = byId.get(String(row[idColumn]));
    const flagged = result ? result.reliability_score < FLAG_THRESHOLD : false;
    if (mode === "cleaned" && flagged) continue;
    out.push({
      ...row,
      reliability_score: result ? result.reliability_score : "",
      flag_reason: result ? result.flag_reason : "",
    });
  }
  return out;
}

function escapeField(value) {
  const s = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function toCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const lines = [headers.map(escapeField).join(",")];
  for (const row of rows) {
    lines.push(headers.map((h) => escapeField(row[h])).join(","));
  }
  return lines.join("\n");
}

export function downloadCsv(filename, csvText) {
  const blob = new Blob([csvText], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
