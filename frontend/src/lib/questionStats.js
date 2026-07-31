/**
 * Rebuild the individual answers from a question's counts.
 *
 * Boxplots need the values themselves rather than a tally, and the counts hold
 * exactly the same information -- {"3": 2} simply means two people answered 3.
 */
export function expandCounts(counts) {
  const values = [];
  for (const point of Object.keys(counts).sort((a, b) => Number(a) - Number(b))) {
    for (let i = 0; i < counts[point]; i += 1) values.push(Number(point));
  }
  return values;
}

/** Turn a question's counts into ordered rows for the distribution chart. */
export function toDistribution(question, mode = "all") {
  const counts = mode === "trustworthy" ? question.counts_trustworthy : question.counts;
  const points = Object.keys(counts).sort((a, b) => Number(a) - Number(b));
  const total = points.reduce((sum, p) => sum + counts[p], 0);
  return points.map((value) => ({
    value,
    count: counts[value],
    pct: total ? (counts[value] / total) * 100 : 0,
  }));
}

function share(counts) {
  const total = Object.values(counts).reduce((s, c) => s + c, 0);
  if (!total) return {};
  return Object.fromEntries(
    Object.entries(counts).map(([point, count]) => [point, count / total])
  );
}

/**
 * A traffic light for a single question.
 *
 * "concern" comes straight from the backend's own note. "watch" is raised here
 * when removing careless responses noticeably changes the answer profile --
 * a sign the question's results were being distorted by them.
 */
export function questionHealth(question) {
  if (question.concern) {
    return { level: "concern", note: question.concern };
  }

  const all = share(question.counts);
  const trustworthy = share(question.counts_trustworthy);
  const biggestShift = Object.keys(all).reduce((max, point) => {
    const delta = Math.abs((all[point] ?? 0) - (trustworthy[point] ?? 0));
    return Math.max(max, delta);
  }, 0);

  if (biggestShift > 0.15) {
    return {
      level: "watch",
      note:
        "Answers shift noticeably once flagged responses are removed, so " +
        "careless responding was skewing this question's results.",
    };
  }
  return { level: "ok", note: "Answers look well spread and stable." };
}

/** Compare the newest run against the oldest, for the trend headline. */
export function summariseHistory(runs) {
  if (!runs.length) {
    return { latest: null, first: null, change: null, direction: "none" };
  }
  const ordered = [...runs].sort(
    (a, b) => new Date(a.recorded_at) - new Date(b.recorded_at)
  );
  const first = ordered[0].overall_quality_pct;
  const latest = ordered[ordered.length - 1].overall_quality_pct;

  if (ordered.length < 2) {
    return { latest, first, change: null, direction: "none" };
  }
  const change = Math.round((latest - first) * 10) / 10;
  return {
    latest,
    first,
    change,
    direction: change > 0 ? "up" : change < 0 ? "down" : "flat",
  };
}
