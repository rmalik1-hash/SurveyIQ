import { useState } from "react";

export default function UploadStep({ onFileReady, onTrySample, busy, error }) {
  const [file, setFile] = useState(null);

  return (
    <section className="card">
      <h2>Upload a survey export</h2>
      <p className="hint">
        Choose the CSV or Excel file your survey tool produced. Nothing is stored on the
        server &mdash; it is analyzed and discarded.
      </p>

      <input
        type="file"
        accept=".csv,.xlsx,.xlsm,text/csv"
        aria-label="Survey file"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />

      {file && <p className="filename">Selected: {file.name}</p>}
      {error && <p className="error">{error}</p>}

      <div className="actions">
        <button className="primary" disabled={!file || busy} onClick={() => onFileReady(file)}>
          {busy ? "Reading columns…" : "Continue"}
        </button>
        <button type="button" disabled={busy} onClick={onTrySample}>
          Try a sample survey
        </button>
      </div>
      <p className="hint" style={{ marginTop: 12, marginBottom: 0 }}>
        No file handy? The sample is generated synthetic data &mdash; useful for seeing how
        the report looks.
      </p>
    </section>
  );
}
