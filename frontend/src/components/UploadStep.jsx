import { useState } from "react";

export default function UploadStep({ onFileReady, busy, error }) {
  const [file, setFile] = useState(null);

  return (
    <section className="card">
      <h2>Upload a survey export</h2>
      <p className="hint">
        Choose the CSV your survey tool produced. Nothing is stored on the server &mdash;
        it is analyzed and discarded.
      </p>

      <input
        type="file"
        accept=".csv,text/csv"
        aria-label="Survey CSV file"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />

      {file && <p className="filename">Selected: {file.name}</p>}
      {error && <p className="error">{error}</p>}

      <button disabled={!file || busy} onClick={() => onFileReady(file)}>
        {busy ? "Reading columns…" : "Continue"}
      </button>
    </section>
  );
}
