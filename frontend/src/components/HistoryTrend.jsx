import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import ChartBoundary from "./ChartBoundary.jsx";
import { summariseHistory } from "../lib/questionStats.js";

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

function shortQuestion(label, max = 58) {
  const tags = label.match(/\[[^\]]+\]/g);
  const stripped = label.replace(/\s*\[[^\]]+\]\s*/g, " ").trim();
  const text = tags ? `${tags.join(" ")} ${stripped}` : label;
  return text.length > max ? `${text.slice(0, max - 3)}...` : text;
}

const METRICS = {
  quality: {
    label: "Data quality (% trustworthy)",
    field: "overall_quality_pct",
    axis: "Trustworthy %",
    range: [0, 100],
    suffix: "% trustworthy",
    color: "#1d9e75",
  },
  mean: {
    label: "Average answer",
    field: "mean_response",
    axis: "Mean answer",
    range: null,
    suffix: " average answer",
    color: "#185fa5",
  },
};

function TrendChart({ runs, metric, question }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const ordered = [...runs].sort(
      (a, b) => new Date(a.recorded_at) - new Date(b.recorded_at)
    );

    const perQuestion = metric === "question";
    const spec = perQuestion
      ? {
          axis: "Average answer",
          range: null,
          suffix: " average",
          color: "#7a4fbf",
        }
      : METRICS[metric];

    const y = perQuestion
      ? ordered.map((r) => r.question_means?.[question] ?? null)
      : ordered.map((r) => r[spec.field] ?? null);

    Plotly.newPlot(
      el,
      [{
        type: "scatter",
        mode: "lines+markers",
        x: ordered.map((r) => r.recorded_at),
        y,
        line: { color: spec.color, width: 2 },
        marker: { size: 9, color: spec.color },
        connectgaps: false,
        hovertemplate:
          `%{x|%d %b %Y}<br>%{y}${spec.suffix}<br>%{customdata} responses<extra></extra>`,
        customdata: ordered.map((r) => r.total),
      }],
      {
        margin: { t: 10, b: 44, l: 52, r: 20 },
        height: 260,
        yaxis: spec.range
          ? { title: spec.axis, range: spec.range }
          : { title: spec.axis },
        xaxis: { title: "Survey date" },
      },
      PLOT_CONFIG
    );
    return () => Plotly.purge(el);
  }, [runs, metric, question]);

  return <div ref={ref} />;
}

export default function HistoryTrend({ runs, surveyLabel, onFilterChange, onClear, busy }) {
  const [filter, setFilter] = useState(surveyLabel || "");
  const [metric, setMetric] = useState("quality");
  const [question, setQuestion] = useState("");

  // Every question seen across the recorded runs, so one can be trended.
  const questionOptions = useMemo(() => {
    const seen = new Set();
    for (const run of runs) {
      for (const label of Object.keys(run.question_means ?? {})) seen.add(label);
    }
    return [...seen].sort();
  }, [runs]);

  const activeQuestion = question || questionOptions[0] || "";
  const labels = useMemo(
    () => [...new Set(runs.map((r) => r.survey_label))].sort(),
    [runs]
  );
  const shown = filter ? runs.filter((r) => r.survey_label === filter) : runs;
  const trend = summariseHistory(
    metric === "question"
      ? shown
          .filter((r) => r.question_means?.[activeQuestion] !== undefined)
          .map((r) => ({
            ...r,
            overall_quality_pct: r.question_means[activeQuestion],
          }))
      : metric === "mean"
        ? shown.map((r) => ({ ...r, overall_quality_pct: r.mean_response }))
        : shown
  );
  const unit = metric === "quality" ? "%" : "";

  if (!runs.length) {
    return (
      <p className="hint" style={{ margin: 0 }}>
        No past analyses recorded yet. Name a survey before analysing it and each run
        will be tracked here, so you can watch data quality change over time.
      </p>
    );
  }

  return (
    <div>
      <div className="explorer-controls">
        <label>
          Survey
          <select
            aria-label="Filter trend by survey"
            value={filter}
            onChange={(e) => {
              setFilter(e.target.value);
              onFilterChange?.(e.target.value);
            }}
          >
            <option value="">All surveys</option>
            {labels.map((l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </label>
        <label>
          Track
          <select
            aria-label="Which measure to trend"
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            {Object.entries(METRICS).map(([key, spec]) => (
              <option key={key} value={key}>{spec.label}</option>
            ))}
            {questionOptions.length > 0 && (
              <option value="question">A specific question</option>
            )}
          </select>
        </label>

        {metric === "question" && (
          <label>
            Question
            <select
              aria-label="Choose a question to trend"
              value={activeQuestion}
              onChange={(e) => setQuestion(e.target.value)}
            >
              {questionOptions.map((label) => (
                <option key={label} value={label}>{shortQuestion(label)}</option>
              ))}
            </select>
          </label>
        )}

        <button type="button" onClick={onClear} disabled={busy}>
          Clear history
        </button>
      </div>

      {trend.direction === "none" ? (
        <p className="hint">
          {shown.length} run recorded. Analyse this survey again later to see a trend.
        </p>
      ) : (
        <div className={`health health-${trend.direction === "down" ? "concern" : "ok"}`}>
          <strong>
            {trend.direction === "up" && `Up ${trend.change}${unit}.`}
            {trend.direction === "down" && `Down ${Math.abs(trend.change)}${unit}.`}
            {trend.direction === "flat" && "No change."}
          </strong>{" "}
          {metric === "question"
            ? `"${shortQuestion(activeQuestion)}" moved from ${trend.first} to ${trend.latest} on average`
            : metric === "mean"
              ? `The average answer moved from ${trend.first} to ${trend.latest}`
              : `Data quality went from ${trend.first}% to ${trend.latest}% trustworthy`}{" "}
          across {shown.length} analyses.
        </div>
      )}

      <ChartBoundary>
        <TrendChart runs={shown} metric={metric} question={activeQuestion} />
      </ChartBoundary>

      <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
        Only totals are stored — the date, survey name, response count, quality score and
        the average answer per question. No individual answers, respondent IDs or
        demographics are ever saved.
      </p>
    </div>
  );
}
