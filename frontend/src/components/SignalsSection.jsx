const SIGNALS = [
  {
    name: "Straightlining",
    desc: "The same answer to every question. Fast to spot, and rarely genuine.",
  },
  {
    name: "Speeding",
    desc: "Finished far quicker than anyone could actually read the questions.",
  },
  {
    name: "Random answering",
    desc: "Answers swing with no coherent pattern — the respondent has checked out.",
  },
  {
    name: "Contradictions",
    desc: "Opposite answers on question pairs that should disagree with each other.",
  },
  {
    name: "Attention checks",
    desc: "Missed the questions written specifically to catch autopilot responding.",
  },
  {
    name: "Extreme responding",
    desc: "Only ever the highest or lowest option, never the middle of the scale.",
  },
];

export default function SignalsSection() {
  return (
    <section className="signals">
      <p className="eyebrow">What we detect</p>
      <h2 className="section-title">Six signals. One score.</h2>

      <div className="signal-grid">
        {SIGNALS.map((s, i) => (
          <div className="signal" key={s.name}>
            <div className="signal-head">
              <span className="signal-num" aria-hidden="true">{i + 1}</span>
              <h3 className="signal-name">{s.name}</h3>
            </div>
            <p className="signal-desc">{s.desc}</p>
          </div>
        ))}
      </div>

      <div className="callout">
        <div>
          <p className="callout-title">Every flag is explainable</p>
          <p className="callout-sub">
            A decision tree, not a black box. Administrators see the actual rule that
            triggered each flag — no respondent is ever flagged without a reason.
          </p>
        </div>
        <div className="tags">
          {["scikit-learn", "FastAPI", "React"].map((t) => (
            <span className="tag" key={t}>{t}</span>
          ))}
        </div>
      </div>
    </section>
  );
}
