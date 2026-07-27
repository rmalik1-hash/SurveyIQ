import { useEffect, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { buildCleanedRows, toCsv, downloadCsv, FLAG_THRESHOLD } from "../lib/cleanedCsv.js";
import ChartBoundary from "./ChartBoundary.jsx";

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

/**
 * Survey headers are long sentences, so axis labels get clipped and unreadable.
 * Most exports carry a short tag like "[Q1]" or "[AC2]" -- prefer that, and fall
 * back to a truncated header. The full label still shows on hover.
 */
function shorten(label, max = 30) {
  const tags = label.match(/\[[^\]]+\]/g);
  if (tags) return tags.join(" / ");
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

function QualityGauge({ pct }) {
  const ref = useRef(null);
  useEffect(() => {
    // Capture the node: by cleanup time React has already nulled ref.current,
    // and Plotly.purge(null) throws.
    const el = ref.current;
    if (!el) return undefined;
    Plotly.newPlot(
      el,
      [{
        type: "indicator",
        mode: "gauge+number",
        value: pct,
        number: { suffix: "%" },
        gauge: {
          axis: { range: [0, 100] },
          bar: { color: "#1d9e75" },
          steps: [
            { range: [0, 50], color: "#fcebeb" },
            { range: [50, 80], color: "#faeeda" },
            { range: [80, 100], color: "#e1f5ee" },
          ],
        },
      }],
      { margin: { t: 10, b: 10, l: 30, r: 30 }, height: 220 },
      PLOT_CONFIG
    );
    return () => Plotly.purge(el);
  }, [pct]);
  return <div ref={ref} />;
}

function QuestionChart({ stats }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ordered = [...stats].sort((a, b) => a.affected - b.affected);
    Plotly.newPlot(
      el,
      [{
        type: "bar",
        orientation: "h",
        x: ordered.map((s) => s.affected),
        y: ordered.map((s) => shorten(s.label)),
        marker: { color: "#378add" },
        hovertext: ordered.map((s) => s.label),
        hoverinfo: "text+x",
      }],
      {
        margin: { t: 10, b: 40, l: 150, r: 20 },
        height: Math.max(160, ordered.length * 42 + 80),
        xaxis: { title: "Respondents affected" },
      },
      PLOT_CONFIG
    );
    return () => Plotly.purge(el);
  }, [stats]);
  return <div ref={ref} />;
}

export default function ResultsDashboard({ result, originalRows, idColumn, fileName, onStartOver }) {
  const { summary, respondents, question_stats: questionStats } = result;
  const flagged = respondents.filter((r) => r.reliability_score < FLAG_THRESHOLD);
  const [showAll, setShowAll] = useState(false);
  const PREVIEW_COUNT = 10;
  const visible = showAll ? flagged : flagged.slice(0, PREVIEW_COUNT);

  const download = (mode) => {
    const rows = buildCleanedRows(originalRows, respondents, idColumn, mode);
    const base = fileName.replace(/\.csv$/i, "");
    downloadCsv(`${base}_${mode}.csv`, toCsv(rows));
  };

  return (
    <section>
      <div className="cards">
        <div className="metric">
          <span className="label">Overall quality</span>
          <span className="value good">{summary.overall_quality_pct}%</span>
        </div>
        <div className="metric">
          <span className="label">Total responses</span>
          <span className="value">{summary.total}</span>
        </div>
        <div className="metric">
          <span className="label">Trustworthy</span>
          <span className="value good">{summary.reliable}</span>
        </div>
        <div className="metric">
          <span className="label">Flagged</span>
          <span className="value bad">{summary.flagged}</span>
        </div>
      </div>

      <div className="card">
        <h2>Data quality</h2>
        <p className="hint">Share of responses the model considers trustworthy.</p>
        <ChartBoundary>
          <QualityGauge pct={summary.overall_quality_pct} />
        </ChartBoundary>
      </div>

      <div className="card">
        <h2>Flagged respondents</h2>
        <p className="hint">Every flag comes with the rule that triggered it.</p>
        {flagged.length === 0 ? (
          <p>No responses were flagged.</p>
        ) : (
          <table className="results-table">
            <thead>
              <tr><th>Respondent</th><th>Reliability</th><th>Why it was flagged</th></tr>
            </thead>
            <tbody>
              {visible.map((r) => (
                <tr key={r.respondent_id}>
                  <td className="mono">{r.respondent_id}</td>
                  <td className="mono">{r.reliability_score.toFixed(2)}</td>
                  <td>{r.flag_reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {flagged.length > PREVIEW_COUNT && (
          <div className="table-foot">
            <span className="hint" style={{ margin: 0 }}>
              Showing {visible.length} of {flagged.length} flagged
            </span>
            <button type="button" onClick={() => setShowAll(!showAll)}>
              {showAll ? "Show fewer" : `Show all ${flagged.length}`}
            </button>
          </div>
        )}
      </div>

      {questionStats.length > 0 && (
        <div className="card">
          <h2>Most problematic questions</h2>
          <p className="hint">Where contradictions and missed attention checks concentrated.</p>
          <ChartBoundary>
            <QuestionChart stats={questionStats} />
          </ChartBoundary>
        </div>
      )}

      <div className="card">
        <h2>Download results</h2>
        <p className="hint">
          <strong>Marked</strong> keeps every response and adds the score and reason.{" "}
          <strong>Cleaned</strong> removes the flagged responses.
        </p>
        <div className="actions">
          <button type="button" onClick={() => download("marked")}>Download marked CSV</button>
          <button type="button" onClick={() => download("cleaned")}>Download cleaned CSV</button>
          <button type="button" onClick={onStartOver}>Analyze another survey</button>
        </div>
      </div>
    </section>
  );
}
