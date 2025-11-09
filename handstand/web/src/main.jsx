import React, { useState } from "react";
import { createRoot } from "react-dom/client";

function App() {
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [feedback, setFeedback] = useState("");
  const [llm, setLlm] = useState("gemini");
  const [model, setModel] = useState("gemini-2.5-pro");
  const [log, setLog] = useState([]);

  const appendLog = (m) => setLog((prev) => [...prev, m]);

  const submit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setResult(null);
    setLog([]);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("llm", llm);
      form.append("gemini_model", model);
      appendLog("Uploading and analyzing...");
      const res = await fetch("/api/analyze-sync", { method: "POST", body: form });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const json = await res.json();
      setResult(json);
      try {
        const fb = await fetch(json.feedback);
        if (fb.ok) {
          const text = await fb.text();
          setFeedback(text);
        } else {
          appendLog(`Feedback fetch failed: ${fb.status}`);
          setFeedback("");
        }
      } catch (e) {
        appendLog(String(e));
        setFeedback("");
      }
      appendLog("Analysis complete.");
    } catch (err) {
      appendLog(String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ maxWidth: 900, margin: "20px auto", fontFamily: "Inter, system-ui, Arial" }}>
      <h1>Handstand Coach</h1>
      <form onSubmit={submit} style={{ border: "1px solid #ddd", padding: 16, borderRadius: 8 }}>
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
          <label>
            LLM:
            <select value={llm} onChange={(e) => setLlm(e.target.value)} style={{ marginLeft: 8 }}>
              <option value="gemini">Gemini</option>
              <option value="openai">OpenAI</option>
            </select>
          </label>
          {llm === "gemini" && (
            <label>
              Model:
              <input
                value={model}
                onChange={(e) => setModel(e.target.value)}
                style={{ marginLeft: 8 }}
              />
            </label>
          )}
          <button type="submit" disabled={!file || busy}>
            {busy ? "Processing..." : "Analyze"}
          </button>
        </div>
      </form>

      <div style={{ marginTop: 16 }}>
        <h3>Logs</h3>
        <pre style={{ background: "#f7f7f7", padding: 12, borderRadius: 8, whiteSpace: "pre-wrap" }}>
          {log.join("\n")}
        </pre>
      </div>

      {result && (
        <div style={{ marginTop: 16 }}>
          <h3>Results</h3>
          <div style={{ marginTop: 8 }}>
            <h4>2D Annotated</h4>
            <video src={result.two_d} width="100%" controls />
          </div>
          <div style={{ marginTop: 12 }}>
            <h4>Feedback</h4>
            {feedback ? (
              <div style={{ whiteSpace: "pre-wrap", background: "#f7f7f7", padding: 12, borderRadius: 8 }}>
                {feedback}
              </div>
            ) : (
              <a href={result.feedback} target="_blank" rel="noreferrer">Open feedback.md</a>
            )}
          </div>
          <div style={{ marginTop: 8 }}>
            <h4>Metrics JSON</h4>
            <a href={result.metrics} target="_blank" rel="noreferrer">View metrics</a>
          </div>
          <div style={{ marginTop: 8, opacity: 0.6 }}>
            <small>
              Advanced assets available for download:&nbsp;
              <a href={result.three_d} target="_blank" rel="noreferrer">3D (axes)</a>,&nbsp;
              <a href={result.three_d_debug} target="_blank" rel="noreferrer">3D Debug</a>
            </small>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);


