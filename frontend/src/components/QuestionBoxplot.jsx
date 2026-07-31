import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import ChartBoundary from "./ChartBoundary.jsx";
import { expandCounts } from "../lib/questionStats.js";

const PLOT_CONFIG = { displayModeBar: false, responsive: true };

function tagFor(label, position) {
  const tags = label.match(/\[[^\]]+\]/g);
  return tags ? tags.join(" ") : `Q${position}`;
}

function Boxes({ questions, mode }) {
  const ref = useRef(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;

    const traces = questions.map((q) => ({
      type: "box",
      y: expandCounts(mode === "trustworthy" ? q.counts_trustworthy : q.counts),
      name: tagFor(q.label, q.position),
      marker: { color: "#378add" },
      line: { color: "#185fa5" },
      boxmean: true,          // show the mean alongside the median
      hovertext: q.label,
    }));

    Plotly.newPlot(
      el,
      traces,
      {
        margin: { t: 10, b: 60, l: 48, r: 20 },
        height: 330,
        showlegend: false,
        yaxis: { title: "Answer", zeroline: false },
        xaxis: { title: "Question", tickangle: -45 },
      },
      PLOT_CONFIG
    );
    return () => Plotly.purge(el);
  }, [questions, mode]);

  return <div ref={ref} />;
}

export default function QuestionBoxplot({ questions, mode }) {
  if (!questions.length) return null;
  return (
    <ChartBoundary>
      <Boxes questions={questions} mode={mode} />
    </ChartBoundary>
  );
}
