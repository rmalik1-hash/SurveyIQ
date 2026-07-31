import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import QuestionExplorer from "../src/components/QuestionExplorer.jsx";
import HistoryTrend from "../src/components/HistoryTrend.jsx";

// Plotly needs a real canvas; the charts themselves are covered by running the
// app, so stub it out and assert on the surrounding behaviour.
vi.mock("plotly.js-dist-min", () => ({
  default: { newPlot: vi.fn(() => Promise.resolve()), purge: vi.fn() },
}));

const questions = [
  {
    label: "I enjoy school [Q1]",
    position: 1,
    counts: { 1: 2, 2: 4, 3: 10, 4: 3, 5: 1 },
    counts_trustworthy: { 1: 1, 2: 4, 3: 9, 4: 3, 5: 1 },
    mean: 2.8, median: 3, mode: 3, concern: null,
  },
  {
    label: "I dislike school [Q2]",
    position: 2,
    counts: { 1: 0, 2: 0, 3: 20, 4: 0, 5: 0 },
    counts_trustworthy: { 1: 0, 2: 0, 3: 18, 4: 0, 5: 0 },
    mean: 3.0, median: 3, mode: 3,
    concern: "Every respondent gave the same answer.",
  },
];

describe("QuestionExplorer", () => {
  it("opens on the boxplot, since it works for any question", () => {
    render(<QuestionExplorer questions={questions} />);
    expect(screen.getByLabelText(/choose a chart/i).value).toBe("boxplot");
    expect(screen.getByText(/middle half of the answers/i)).toBeInTheDocument();
  });

  it("shows mean, median and mode for a single question", () => {
    render(<QuestionExplorer questions={questions} />);
    fireEvent.change(screen.getByLabelText(/choose a chart/i), {
      target: { value: "distribution" },
    });
    expect(screen.getByText("mean")).toBeInTheDocument();
    expect(screen.getByText("median")).toBeInTheDocument();
    expect(screen.getByText("mode")).toBeInTheDocument();
  });

  it("lets the user pick a different question", () => {
    render(<QuestionExplorer questions={questions} />);
    fireEvent.change(screen.getByLabelText(/choose a chart/i), {
      target: { value: "distribution" },
    });
    fireEvent.change(screen.getByLabelText(/choose a question/i), { target: { value: "1" } });
    // Q2's concern should now be surfaced
    expect(screen.getByText(/same answer/i)).toBeInTheDocument();
  });

  it("can restrict the view to trustworthy respondents", () => {
    render(<QuestionExplorer questions={questions} />);
    const toggle = screen.getByLabelText(/which respondents/i);
    fireEvent.change(toggle, { target: { value: "trustworthy" } });
    expect(toggle.value).toBe("trustworthy");
  });

  it("says so plainly when there are no scale questions", () => {
    render(<QuestionExplorer questions={[]} />);
    expect(screen.getByText(/nothing to break down/i)).toBeInTheDocument();
  });
});

const runs = [
  {
    recorded_at: "2026-01-01T10:00:00+00:00", survey_label: "Wellbeing",
    total: 100, flagged: 40, reliable: 60, overall_quality_pct: 60,
    mean_response: 3.0, question_means: { "I like this class [Q1]": 3.0 },
  },
  {
    recorded_at: "2026-02-01T10:00:00+00:00", survey_label: "Wellbeing",
    total: 100, flagged: 25, reliable: 75, overall_quality_pct: 75,
    mean_response: 3.4, question_means: { "I like this class [Q1]": 4.2 },
  },
];

describe("HistoryTrend", () => {
  it("explains how to start tracking when nothing is recorded", () => {
    render(<HistoryTrend runs={[]} />);
    expect(screen.getByText(/no past analyses recorded yet/i)).toBeInTheDocument();
  });

  it("reports the direction of travel", () => {
    render(<HistoryTrend runs={runs} />);
    expect(screen.getByText(/up 15%/i)).toBeInTheDocument();
    expect(screen.getByText(/60% to 75% trustworthy/i)).toBeInTheDocument();
  });

  it("can trend a single question's average answer", () => {
    render(<HistoryTrend runs={runs} />);
    fireEvent.change(screen.getByLabelText(/which measure to trend/i), {
      target: { value: "question" },
    });
    // the question picker only appears for this measure
    const picker = screen.getByLabelText(/choose a question to trend/i);
    expect(picker).toBeInTheDocument();
    // Q1 climbed 3.0 -> 4.2 across the two runs
    expect(screen.getByText(/moved from 3 to 4.2 on average/i)).toBeInTheDocument();
  });

  it("has no question picker until questions have been recorded", () => {
    const bare = runs.map(({ question_means, ...rest }) => rest);
    render(<HistoryTrend runs={bare} />);
    expect(screen.queryByLabelText(/choose a question to trend/i)).toBeNull();
  });

  it("does not claim a trend from a single run", () => {
    render(<HistoryTrend runs={[runs[0]]} />);
    expect(screen.getByText(/analyse this survey again later/i)).toBeInTheDocument();
  });

  it("can switch between quality and average answer", () => {
    render(<HistoryTrend runs={runs} />);
    const metric = screen.getByLabelText(/which measure to trend/i);
    fireEvent.change(metric, { target: { value: "mean" } });
    expect(metric.value).toBe("mean");
  });

  it("states exactly what gets stored", () => {
    render(<HistoryTrend runs={runs} />);
    expect(
      screen.getByText(/no individual answers, respondent IDs or demographics/i)
    ).toBeInTheDocument();
    // the disclosure must name per-question averages, which are also stored
    expect(screen.getByText(/average answer per question/i)).toBeInTheDocument();
  });

  it("offers a way to clear the history", () => {
    const onClear = vi.fn();
    render(<HistoryTrend runs={runs} onClear={onClear} />);
    fireEvent.click(screen.getByRole("button", { name: /clear history/i }));
    expect(onClear).toHaveBeenCalled();
  });
});
