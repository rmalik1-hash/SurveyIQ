import { useState } from "react";
import Papa from "papaparse";
import { fetchColumns, analyzeSurvey, generateSampleSurvey } from "./lib/api.js";
import UploadStep from "./components/UploadStep.jsx";
import MappingStep from "./components/MappingStep.jsx";
import ResultsDashboard from "./components/ResultsDashboard.jsx";
import "./App.css";

function parseCsv(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data),
      error: reject,
    });
  });
}

export default function App() {
  const [step, setStep] = useState("upload");
  const [file, setFile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [rows, setRows] = useState([]);
  const [idColumn, setIdColumn] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const handleFile = async (chosen) => {
    setBusy(true);
    setError("");
    try {
      // Ask the API for the columns and parse the file locally. The local copy
      // is what lets us build the cleaned CSV later without re-uploading.
      const [cols, parsed] = await Promise.all([fetchColumns(chosen), parseCsv(chosen)]);
      setFile(chosen);
      setColumns(cols);
      setRows(parsed);
      setStep("map");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleTrySample = async () => {
    setBusy(true);
    setError("");
    try {
      await handleFile(await generateSampleSurvey());
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  };

  const handleAnalyze = async (mapping) => {
    setBusy(true);
    setError("");
    try {
      const body = await analyzeSurvey(file, mapping);
      const idCol = Object.entries(mapping.columns).find(([, r]) => r === "respondent_id")[0];
      setIdColumn(idCol);
      setResult(body);
      setStep("results");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const startOver = () => {
    setStep("upload");
    setFile(null);
    setColumns([]);
    setRows([]);
    setResult(null);
    setError("");
  };

  return (
    <div className="wrap">
      <header className="topbar">
        <div className="mark">SQ</div>
        <div>
          <h1>SurveyIQ</h1>
          <p className="hint">Which survey responses can you trust?</p>
        </div>
      </header>

      {step === "upload" && (
        <UploadStep
          onFileReady={handleFile}
          onTrySample={handleTrySample}
          busy={busy}
          error={error}
        />
      )}
      {step === "map" && (
        <MappingStep
          columns={columns}
          onAnalyze={handleAnalyze}
          onBack={startOver}
          busy={busy}
          error={error}
        />
      )}
      {step === "results" && result && (
        <ResultsDashboard
          result={result}
          originalRows={rows}
          idColumn={idColumn}
          fileName={file?.name ?? "survey.csv"}
          onStartOver={startOver}
        />
      )}
    </div>
  );
}
