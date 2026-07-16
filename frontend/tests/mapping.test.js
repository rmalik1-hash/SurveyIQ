import { describe, it, expect } from "vitest";
import { buildMapping, validateMapping } from "../src/lib/mapping.js";

const baseState = () => ({
  roles: {
    "Response ID": "respondent_id",
    "Start Time": "start_time",
    "Timestamp": "end_time",
    "Q1": "question",
    "Q2": "question",
    "AC1": "attention_check",
    "Email Address": "ignore",
    "grade level": "demographic",
  },
  scaleMin: 1,
  scaleMax: 5,
  acAnswers: { AC1: 5 },
  pairs: [],
});

describe("buildMapping", () => {
  it("produces the mapping shape the API expects", () => {
    const m = buildMapping(baseState());
    expect(m.columns["Response ID"]).toBe("respondent_id");
    expect(m.columns["Q1"]).toBe("question");
    expect(m.columns["Email Address"]).toBe("ignore");
    expect(m.scale).toEqual([1, 5]);
    expect(m.attention_check_answers).toEqual({ AC1: 5 });
  });

  it("omits contradiction_pairs when there are none", () => {
    expect(buildMapping(baseState()).contradiction_pairs).toBeUndefined();
  });

  it("includes complete contradiction pairs only", () => {
    const state = { ...baseState(), pairs: [["Q1", "Q2"], ["Q1", ""]] };
    expect(buildMapping(state).contradiction_pairs).toEqual([["Q1", "Q2"]]);
  });

  it("coerces string inputs to numbers", () => {
    const state = { ...baseState(), scaleMin: "1", scaleMax: "7", acAnswers: { AC1: "7" } };
    const m = buildMapping(state);
    expect(m.scale).toEqual([1, 7]);
    expect(m.attention_check_answers).toEqual({ AC1: 7 });
  });
});

describe("validateMapping", () => {
  it("accepts a valid state", () => {
    expect(validateMapping(baseState())).toEqual([]);
  });

  it("requires exactly one respondent_id", () => {
    const none = { ...baseState(), roles: { ...baseState().roles, "Response ID": "ignore" } };
    expect(validateMapping(none).length).toBeGreaterThan(0);
    const two = { ...baseState(), roles: { ...baseState().roles, "grade level": "respondent_id" } };
    expect(validateMapping(two).length).toBeGreaterThan(0);
  });

  it("requires at least one question", () => {
    const roles = { ...baseState().roles, Q1: "demographic", Q2: "demographic" };
    expect(validateMapping({ ...baseState(), roles }).length).toBeGreaterThan(0);
  });

  it("requires scale min < max", () => {
    expect(validateMapping({ ...baseState(), scaleMin: 5, scaleMax: 1 }).length).toBeGreaterThan(0);
  });

  it("requires a correct answer for each attention check", () => {
    expect(validateMapping({ ...baseState(), acAnswers: {} }).length).toBeGreaterThan(0);
    expect(validateMapping({ ...baseState(), acAnswers: { AC1: "" } }).length).toBeGreaterThan(0);
  });

  it("requires contradiction pairs to reference question columns", () => {
    const state = { ...baseState(), pairs: [["Q1", "grade level"]] };
    expect(validateMapping(state).length).toBeGreaterThan(0);
  });

  it("rejects a pair of the same question", () => {
    expect(validateMapping({ ...baseState(), pairs: [["Q1", "Q1"]] }).length).toBeGreaterThan(0);
  });
});
