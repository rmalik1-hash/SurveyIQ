import { describe, it, expect } from "vitest";
import {
  toDistribution,
  questionHealth,
  summariseHistory,
} from "../src/lib/questionStats.js";

const question = {
  label: "How much do you agree? [Q1]",
  position: 1,
  counts: { 1: 2, 2: 4, 3: 10, 4: 3, 5: 1 },
  counts_trustworthy: { 1: 1, 2: 4, 3: 9, 4: 3, 5: 1 },
  mean: 2.8,
  concern: null,
};

describe("toDistribution", () => {
  it("returns one entry per scale point, in scale order", () => {
    const rows = toDistribution(question, "all");
    expect(rows.map((r) => r.value)).toEqual(["1", "2", "3", "4", "5"]);
    expect(rows.map((r) => r.count)).toEqual([2, 4, 10, 3, 1]);
  });

  it("computes each point's share of responses", () => {
    const rows = toDistribution(question, "all");
    expect(rows[2].pct).toBeCloseTo(50);  // 10 of 20
    expect(rows.reduce((s, r) => s + r.pct, 0)).toBeCloseTo(100);
  });

  it("can show trustworthy respondents only", () => {
    const rows = toDistribution(question, "trustworthy");
    expect(rows.map((r) => r.count)).toEqual([1, 4, 9, 3, 1]);
  });

  it("handles a question nobody answered without dividing by zero", () => {
    const empty = { ...question, counts: { 1: 0, 2: 0, 3: 0 } };
    const rows = toDistribution(empty, "all");
    expect(rows.every((r) => r.pct === 0)).toBe(true);
  });
});

describe("questionHealth", () => {
  it("passes a question with a healthy spread", () => {
    expect(questionHealth(question).level).toBe("ok");
  });

  it("reports a concern raised by the backend", () => {
    const flat = { ...question, concern: "Every respondent gave the same answer." };
    const health = questionHealth(flat);
    expect(health.level).toBe("concern");
    expect(health.note).toContain("same answer");
  });

  it("notices when trustworthy answers differ markedly from the whole sample", () => {
    const skewed = {
      ...question,
      counts: { 1: 10, 2: 0, 3: 0, 4: 0, 5: 10 },
      counts_trustworthy: { 1: 0, 2: 0, 3: 0, 4: 0, 5: 10 },
    };
    expect(questionHealth(skewed).level).toBe("watch");
  });
});

describe("summariseHistory", () => {
  const runs = [
    { recorded_at: "2026-01-01T10:00:00+00:00", survey_label: "W", total: 100, flagged: 40, reliable: 60, overall_quality_pct: 60 },
    { recorded_at: "2026-02-01T10:00:00+00:00", survey_label: "W", total: 100, flagged: 25, reliable: 75, overall_quality_pct: 75 },
  ];

  it("reports the change between the first and latest run", () => {
    const s = summariseHistory(runs);
    expect(s.latest).toBe(75);
    expect(s.change).toBe(15);
    expect(s.direction).toBe("up");
  });

  it("calls a decline a decline", () => {
    const s = summariseHistory([runs[1], { ...runs[0], recorded_at: "2026-03-01T10:00:00+00:00" }]);
    expect(s.direction).toBe("down");
    expect(s.change).toBe(-15);
  });

  it("needs at least two runs before claiming a trend", () => {
    expect(summariseHistory([runs[0]]).direction).toBe("none");
    expect(summariseHistory([]).latest).toBe(null);
  });
});
