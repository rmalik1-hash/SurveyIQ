import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import ChartBoundary from "./ChartBoundary.jsx";
import { summariseHistory } from "../lib/questionStats.js";

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

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

function TrendChart({ runs, metric }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const spec = METRICS[metric];
    const ordered = [...runs].sort(
      (a, b) => new Date(a.recorded_at) - new Date(b.recorded_at)
    );
    Plotly.newPlot(
      el,
      [{
        type: "scatter",
        mode: "lines+markers",
        x: ordered.map((r) => r.recorded_at),
        y: ordered.map((r) => r[spec.field] ?? null),
        line: { color: spec.color, width: 2 },
        marker: { size: 9, color: spec.color },
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
        xaxis: { title: "Analysed on" },
      },
      PLOT_CONFIG
    );
    return () => Plotly.purge(el);
  }, [runs, metric]);

  return <div ref={ref} />;
}

export default function HistoryTrend({ runs, surveyLabel, onFilterChange, onClear, busy }) {
  const [filter, setFilter] = useState(surveyLabel || "");
  const [metric, setMetric] = useState("quality");
  const labels = useMemo(
    () => [...new Set(runs.map((r) => r.survey_label))].sort(),
    [runs]
  );
  const shown = filter ? runs.filter((r) => r.survey_label === filter) : runs;
  const trend = summariseHistory(shown);

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
          </select>
        </label>

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
            {trend.direction === "up" && `Up ${trend.change} points.`}
            {trend.direction === "down" && `Down ${Math.abs(trend.change)} points.`}
            {trend.direction === "flat" && "No change."}
          </strong>{" "}
          Data quality went from {trend.first}% to {trend.latest}% trustworthy across{" "}
          {shown.length} analyses.
        </div>
      )}

      <ChartBoundary>
        <TrendChart runs={shown} metric={metric} />
      </ChartBoundary>

      <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
        Only these totals are stored — the date, survey name, response count and quality
        score. No answers, respondent IDs or demographics are ever saved.
      </p>
    </div>
  );
}
