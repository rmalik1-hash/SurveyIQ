import { useMemo, useState } from "react";
import { ROLES, ROLE_LABELS, buildMapping, validateMapping } from "../lib/mapping.js";

function guessRole(column) {
  const c = column.toLowerCase();
  if (column.includes("[Q")) return "question";
  if (column.includes("[AC")) return "attention_check";
  if (c.includes("response id") || c === "id") return "respondent_id";
  if (c.includes("start")) return "start_time";
  if (c.includes("timestamp") || c.includes("end time")) return "end_time";
  if (c.includes("email") || c.includes("name")) return "ignore";
  return "demographic";
}

export default function MappingStep({ columns, onAnalyze, onBack, busy, error }) {
  const [roles, setRoles] = useState(() =>
    Object.fromEntries(columns.map((c) => [c, guessRole(c)]))
  );
  const [scaleMin, setScaleMin] = useState(1);
  const [scaleMax, setScaleMax] = useState(5);
  const [acAnswers, setAcAnswers] = useState({});
  const [pairs, setPairs] = useState([]);
  const [showErrors, setShowErrors] = useState(false);

  const state = { roles, scaleMin, scaleMax, acAnswers, pairs };
  const questionCols = useMemo(
    () => columns.filter((c) => roles[c] === "question"),
    [columns, roles]
  );
  const acCols = useMemo(
    () => columns.filter((c) => roles[c] === "attention_check"),
    [columns, roles]
  );
  const errors = validateMapping(state);

  const submit = () => {
    setShowErrors(true);
    if (errors.length === 0) onAnalyze(buildMapping(state));
  };

  return (
    <section className="card">
      <h2>Tag your columns</h2>
      <p className="hint">
        Tell SurveyIQ what each column is. We guessed where we could &mdash; correct anything wrong.
      </p>

      <div className="scale-row">
        <label>
          Scale minimum
          <input type="number" value={scaleMin} onChange={(e) => setScaleMin(e.target.value)} />
        </label>
        <label>
          Scale maximum
          <input type="number" value={scaleMax} onChange={(e) => setScaleMax(e.target.value)} />
        </label>
      </div>

      <table className="map-table">
        <thead>
          <tr><th>Column</th><th>Role</th></tr>
        </thead>
        <tbody>
          {columns.map((col) => (
            <tr key={col}>
              <td className="col-name" title={col}>{col}</td>
              <td>
                <select
                  aria-label={`Role for ${col}`}
                  value={roles[col]}
                  onChange={(e) => setRoles({ ...roles, [col]: e.target.value })}
                >
                  {ROLES.map((r) => (
                    <option key={r} value={r}>{ROLE_LABELS[r]}</option>
                  ))}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {acCols.length > 0 && (
        <>
          <h3>Correct answers for attention checks</h3>
          {acCols.map((col) => (
            <label key={col} className="ac-row">
              <span className="col-name" title={col}>{col}</span>
              <input
                type="number"
                aria-label={`Correct answer for ${col}`}
                value={acAnswers[col] ?? ""}
                onChange={(e) => setAcAnswers({ ...acAnswers, [col]: e.target.value })}
              />
            </label>
          ))}
        </>
      )}

      <h3>Reverse-coded question pairs (optional)</h3>
      <p className="hint">
        Pairs of questions that should get opposite answers, e.g. &ldquo;I enjoy school&rdquo; and
        &ldquo;I dislike school&rdquo;. Used to detect contradictions.
      </p>
      {pairs.map(([a, b], i) => (
        <div key={i} className="pair-row">
          <select
            aria-label={`Pair ${i + 1} first question`}
            value={a}
            onChange={(e) => {
              const next = [...pairs];
              next[i] = [e.target.value, b];
              setPairs(next);
            }}
          >
            <option value="">Choose a question…</option>
            {questionCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <select
            aria-label={`Pair ${i + 1} second question`}
            value={b}
            onChange={(e) => {
              const next = [...pairs];
              next[i] = [a, e.target.value];
              setPairs(next);
            }}
          >
            <option value="">Choose a question…</option>
            {questionCols.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <button type="button" onClick={() => setPairs(pairs.filter((_, j) => j !== i))}>
            Remove
          </button>
        </div>
      ))}
      <button type="button" onClick={() => setPairs([...pairs, ["", ""]])}>
        Add a pair
      </button>

      {showErrors && errors.length > 0 && (
        <ul className="error">
          {errors.map((e) => <li key={e}>{e}</li>)}
        </ul>
      )}
      {error && <p className="error">{error}</p>}

      <div className="actions">
        <button type="button" onClick={onBack}>Back</button>
        <button type="button" className="primary" disabled={busy} onClick={submit}>
          {busy ? "Analyzing…" : "Analyze responses"}
        </button>
      </div>
    </section>
  );
}
