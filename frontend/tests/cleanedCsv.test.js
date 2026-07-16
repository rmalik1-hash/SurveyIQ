import { describe, it, expect } from "vitest";
import { buildCleanedRows, toCsv } from "../src/lib/cleanedCsv.js";

const rows = [
  { "Response ID": "R1", Q1: 3, note: "fine" },
  { "Response ID": "R2", Q1: 1, note: "fast" },
];

const respondents = [
  { respondent_id: "R1", reliability_score: 0.95, flag_reason: "" },
  { respondent_id: "R2", reliability_score: 0.1, flag_reason: "Answered very fast (x <= 1)." },
];

describe("buildCleanedRows", () => {
  it("marks every row with score and reason, keeping originals", () => {
    const out = buildCleanedRows(rows, respondents, "Response ID", "marked");
    expect(out).toHaveLength(2);
    expect(out[0]["Response ID"]).toBe("R1");
    expect(out[0].Q1).toBe(3);
    expect(out[0].reliability_score).toBe(0.95);
    expect(out[0].flag_reason).toBe("");
    expect(out[1].reliability_score).toBe(0.1);
    expect(out[1].flag_reason).toContain("Answered very fast");
  });

  it("removes flagged rows in cleaned mode", () => {
    const out = buildCleanedRows(rows, respondents, "Response ID", "cleaned");
    expect(out).toHaveLength(1);
    expect(out[0]["Response ID"]).toBe("R1");
  });

  it("joins by respondent id regardless of row order", () => {
    const reversed = [rows[1], rows[0]];
    const out = buildCleanedRows(reversed, respondents, "Response ID", "marked");
    expect(out[0]["Response ID"]).toBe("R2");
    expect(out[0].reliability_score).toBe(0.1);
  });

  it("leaves score blank for a row with no matching result", () => {
    const extra = [...rows, { "Response ID": "R9", Q1: 2, note: "unknown" }];
    const out = buildCleanedRows(extra, respondents, "Response ID", "marked");
    expect(out).toHaveLength(3);
    expect(out[2].reliability_score).toBe("");
  });
});

describe("toCsv", () => {
  it("writes a header row and values", () => {
    const csv = toCsv([{ a: 1, b: "x" }]);
    expect(csv).toBe("a,b\n1,x");
  });

  it("quotes fields containing commas, quotes, or newlines", () => {
    const csv = toCsv([{ a: "hello, world", b: 'say "hi"' }]);
    expect(csv).toContain('"hello, world"');
    expect(csv).toContain('"say ""hi"""');
  });

  it("returns an empty string for no rows", () => {
    expect(toCsv([])).toBe("");
  });
});
