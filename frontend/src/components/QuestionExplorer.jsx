import { useEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import ChartBoundary from "./ChartBoundary.jsx";
import QuestionBoxplot from "./QuestionBoxplot.jsx";
import { toDistribution, questionHealth } from "../lib/questionStats.js";

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

const HEALTH_LABEL = {
  ok: "Looks healthy",
  watch: "Worth a look",
  concern: "Needs attention",
};

function shortLabel(label, max = 70) {
  const tags = label.match(/\[[^\]]+\]/g);
  if (tags) {
    const stripped = label.replace(/\s*\[[^\]]+\]\s*/g, " ").trim();
    return `${tags.join(" ")} ${stripped}`.slice(0, max);
  }
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

function DistributionChart({ question, mode }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    const rows = toDistribution(question, mode);
    Plotly.newPlot(
      el,
      [{
        type: "bar",
        orientation: "h",
        x: rows.map((r) => r.count),
        y: rows.map((r) => `Answer ${r.value}`),
        marker: { color: "#378add" },
        text: rows.map((r) => `${r.count} (${r.pct.toFixed(0)}%)`),
        textposition: "auto",
        hovertemplate: "%{y}: %{x} respondents<extra></extra>",
      }],
      {
        margin: { t: 10, b: 42, l: 88, r: 24 },
        height: Math.max(190, rows.length * 42 + 70),
        xaxis: { title: "Respondents", rangemode: "tozero" },
        yaxis: { autorange: "reversed" },
      },
      PLOT_CONFIG
    );
    return () => Plotly.purge(el);
  }, [question, mode]);

  return <div ref={ref} />;
}

export default function QuestionExplorer({ questions }) {
  const [selected, setSelected] = useState(0);
  const [mode, setMode] = useState("all");
  const [view, setView] = useState("boxplot");

  const question = questions[selected];
  const health = useMemo(() => (question ? questionHealth(question) : null), [question]);

  if (!questions.length) {
    return <p className="hint">No scale questions were mapped, so there is nothing to break down.</p>;
  }

  const rows = toDistribution(question, mode);
  const answered = rows.reduce((sum, r) => sum + r.count, 0);
  const top = rows.reduce((best, r) => (r.count > best.count ? r : best), rows[0]);

  return (
    <div>
      <div className="explorer-controls">
        <label>
          View
          <select
            aria-label="Choose a chart"
            value={view}
            onChange={(e) => setView(e.target.value)}
          >
            <option value="boxplot">All questions at a glance (boxplot)</option>
            <option value="distribution">One question in detail (distribution)</option>
          </select>
        </label>

        {view === "distribution" && (
          <label>
            Question
            <select
              aria-label="Choose a question"
              value={selected}
              onChange={(e) => setSelected(Number(e.target.value))}
            >
              {questions.map((q, i) => (
                <option key={q.label} value={i}>
                  {i + 1}. {shortLabel(q.label)}
                </option>
              ))}
            </select>
          </label>
        )}

        <label>
          Show
          <select
            aria-label="Which respondents to include"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="all">All respondents</option>
            <option value="trustworthy">Trustworthy only</option>
          </select>
        </label>
      </div>

      {view === "boxplot" ? (
        <>
          <p className="hint">
            Each box covers the middle half of the answers, the line inside is the median
            and the dashed marker is the mean. A tall box means opinion was divided; a flat
            one means everybody answered much the same.
          </p>
          <QuestionBoxplot questions={questions} mode={mode} />
        </>
      ) : (
        <>
          <div className={`health health-${health.level}`}>
            <strong>{HEALTH_LABEL[health.level]}.</strong> {health.note}
          </div>

          <div className="question-facts">
            <span><strong>{answered}</strong> answered</span>
            <span>mean <strong>{question.mean ?? "—"}</strong></span>
            <span>median <strong>{question.median ?? "—"}</strong></span>
            <span>mode <strong>{question.mode ?? "—"}</strong></span>
            <span>most common: <strong>answer {top.value}</strong> ({top.pct.toFixed(0)}%)</span>
          </div>

          <ChartBoundary>
            <DistributionChart question={question} mode={mode} />
          </ChartBoundary>
        </>
      )}
    </div>
  );
}
