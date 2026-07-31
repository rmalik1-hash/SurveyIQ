import { useState } from "react";
import Papa from "papaparse";
import {
  fetchColumns,
  analyzeSurvey,
  generateSampleSurvey,
  fetchHistory,
  clearHistory,
} from "./lib/api.js";
import Navbar from "./components/Navbar.jsx";
import Hero from "./components/Hero.jsx";
import SignalsSection from "./components/SignalsSection.jsx";
import Footer from "./components/Footer.jsx";
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
  const [surveyLabel, setSurveyLabel] = useState("");
  const [history, setHistory] = useState([]);
  const [historyBusy, setHistoryBusy] = useState(false);

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
      const body = await analyzeSurvey(file, mapping, surveyLabel);
      const idCol = Object.entries(mapping.columns).find(([, r]) => r === "respondent_id")[0];
      setIdColumn(idCol);
      setResult(body);
      setStep("results");
      // Past runs power the trend chart. A failure here must not lose the results.
      try {
        setHistory(await fetchHistory());
      } catch {
        setHistory([]);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleClearHistory = async () => {
    setHistoryBusy(true);
    try {
      await clearHistory();
      setHistory([]);
    } catch (e) {
      setError(e.message);
    } finally {
      setHistoryBusy(false);
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
    <div className="page">
      <Navbar step={step} />

      <main className="wrap">
        {step === "upload" && <Hero />}

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
            surveyLabel={surveyLabel}
            onSurveyLabelChange={setSurveyLabel}
          />
        )}

        {step === "results" && result && (
          <ResultsDashboard
            result={result}
            originalRows={rows}
            idColumn={idColumn}
            fileName={file?.name ?? "survey.csv"}
            onStartOver={startOver}
            history={history}
            surveyLabel={surveyLabel}
            onClearHistory={handleClearHistory}
            historyBusy={historyBusy}
          />
        )}

        {step === "upload" && <SignalsSection />}
      </main>

      <Footer />
    </div>
  );
}
