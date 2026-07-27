const STEPS = [
  { id: "upload", label: "Upload" },
  { id: "map", label: "Map columns" },
  { id: "results", label: "Results" },
];

export default function Navbar({ step }) {
  const currentIndex = STEPS.findIndex((s) => s.id === step);

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="brand">
          <div className="mark" aria-hidden="true">SQ</div>
          <span className="brand-name">SurveyIQ</span>
        </div>

        <nav className="steps" aria-label="Progress">
          {STEPS.map((s, i) => {
            const state = i < currentIndex ? "done" : i === currentIndex ? "current" : "todo";
            return (
              <span key={s.id} className="step-item">
                <span
                  className={`step ${state}`}
                  aria-current={state === "current" ? "step" : undefined}
                >
                  {state === "done" && <span aria-hidden="true">✓ </span>}
                  {s.label}
                </span>
                {i < STEPS.length - 1 && <span className="step-sep" aria-hidden="true">—</span>}
              </span>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
