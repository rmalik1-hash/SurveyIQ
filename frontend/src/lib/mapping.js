export const ROLES = [
  "ignore",
  "respondent_id",
  "start_time",
  "end_time",
  "question",
  "attention_check",
  "demographic",
];

export const ROLE_LABELS = {
  ignore: "Ignore",
  respondent_id: "Respondent ID",
  start_time: "Start time",
  end_time: "End time",
  question: "Survey question",
  attention_check: "Attention check",
  demographic: "Demographic",
};

function isFilledNumber(value) {
  return value !== undefined && value !== null && value !== "" && Number.isFinite(Number(value));
}

function completePairs(pairs) {
  return (pairs || []).filter(([a, b]) => a && b);
}

/** Build the exact mapping object the /analyze endpoint expects. */
export function buildMapping({ roles, scaleMin, scaleMax, acAnswers, pairs }) {
  const columns = { ...roles };
  const attention_check_answers = {};
  for (const [col, role] of Object.entries(roles)) {
    if (role === "attention_check") {
      attention_check_answers[col] = Number(acAnswers[col]);
    }
  }
  const mapping = {
    columns,
    scale: [Number(scaleMin), Number(scaleMax)],
    attention_check_answers,
  };
  const valid = completePairs(pairs);
  if (valid.length > 0) mapping.contradiction_pairs = valid;
  return mapping;
}

/**
 * Client-side mirror of the backend's mapping rules, surfaced early for a
 * better experience. The backend remains the authority.
 */
export function validateMapping({ roles, scaleMin, scaleMax, acAnswers, pairs }) {
  const errors = [];
  const assigned = Object.values(roles);

  const idCount = assigned.filter((r) => r === "respondent_id").length;
  if (idCount !== 1) errors.push("Tag exactly one column as the respondent ID.");
  if (!assigned.includes("question")) errors.push("Tag at least one column as a survey question.");

  const min = Number(scaleMin);
  const max = Number(scaleMax);
  if (!isFilledNumber(scaleMin) || !isFilledNumber(scaleMax) || min >= max) {
    errors.push("Scale needs a minimum lower than its maximum.");
  }

  for (const [col, role] of Object.entries(roles)) {
    if (role === "attention_check" && !isFilledNumber(acAnswers[col])) {
      errors.push(`Attention check "${col}" needs its correct answer.`);
    }
  }

  for (const [a, b] of completePairs(pairs)) {
    if (a === b) {
      errors.push("A contradiction pair must use two different questions.");
    } else if (roles[a] !== "question" || roles[b] !== "question") {
      errors.push("Contradiction pairs must use columns tagged as survey questions.");
    }
  }

  return errors;
}
