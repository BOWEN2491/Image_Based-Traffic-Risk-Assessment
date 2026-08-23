import { useEffect, useRef, useState } from "react";

import "./App.css";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const API_URL = `${API_BASE_URL}/api/predict`;
const ALLOWED_RISKS = new Set(["low", "medium", "high", "unknown"]);
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const REQUEST_TIMEOUT_MS = 30_000;

function readErrorMessage(payload, status) {
  const detail = payload?.detail ?? payload?.error?.message ?? payload?.message;

  if (typeof detail?.message === "string" && detail.message.trim()) {
    return detail.message;
  }

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg)
      .filter(Boolean)
      .join("; ");
    if (messages) return messages;
  }

  return status >= 500
    ? "The analysis service is temporarily unavailable. Please try again later."
    : "The selected image could not be processed. Please check the file and try again.";
}

async function readJson(response) {
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) return null;

  try {
    return await response.json();
  } catch {
    return null;
  }
}

function validateResult(payload) {
  const validStatus = payload?.status === "ok" || payload?.status === "uncertain";
  const validRisk = ALLOWED_RISKS.has(payload?.risk);
  const validPair =
    (payload?.status === "uncertain") === (payload?.risk === "unknown");

  if (!validStatus || !validRisk || !validPair) {
    throw new Error("The service returned an unexpected response. Please try again later.");
  }

  return {
    ...payload,
    features:
      payload.features && typeof payload.features === "object" ? payload.features : {},
    model_versions:
      payload.model_versions && typeof payload.model_versions === "object"
        ? payload.model_versions
        : {},
  };
}

function DataTable({ caption, data }) {
  const entries = Object.entries(data);
  if (entries.length === 0) return null;

  return (
    <div className="table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead>
          <tr>
            <th scope="col">Name</th>
            <th scope="col">Value</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, value]) => (
            <tr key={key}>
              <th scope="row">{key}</th>
              <td>{typeof value === "object" ? JSON.stringify(value) : String(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const requestRef = useRef(null);
  const submittingRef = useRef(false);

  useEffect(
    () => () => {
      requestRef.current?.abort();
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl],
  );

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0] ?? null;
    requestRef.current?.abort();
    requestRef.current = null;
    submittingRef.current = false;

    if (selected && (!ALLOWED_TYPES.has(selected.type) || selected.size > MAX_FILE_BYTES)) {
      setFile(null);
      setPreviewUrl(null);
      setError(
        selected.size > MAX_FILE_BYTES
          ? "The image must be 10 MiB or smaller."
          : "Please select a JPEG, PNG, or WebP image.",
      );
      return;
    }
    setFile(selected);
    setPreviewUrl(selected ? URL.createObjectURL(selected) : null);
    setResult(null);
    setError("");
    setLoading(false);
  };

  const cancelAnalysis = () => requestRef.current?.abort();

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file || submittingRef.current) return;

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort("timeout"), REQUEST_TIMEOUT_MS);
    requestRef.current = controller;
    submittingRef.current = true;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      const payload = await readJson(response);

      if (!response.ok) {
        throw new Error(readErrorMessage(payload, response.status));
      }

      setResult(validateResult(payload));
    } catch (requestError) {
      if (requestRef.current === controller) {
        setError(
          requestError.name === "AbortError"
            ? (controller.signal.reason === "timeout" ? "Analysis timed out after 30 seconds." : "Analysis canceled.")
            : requestError.message || "The request failed. Please try again.",
        );
      }
    } finally {
      if (requestRef.current === controller) {
        clearTimeout(timeout);
        requestRef.current = null;
        submittingRef.current = false;
        setLoading(false);
      }
    }
  };

  const isUncertain = result?.status === "uncertain";

  return (
    <main className="app-shell">
      <header className="hero">
        <p className="eyebrow">Research demo</p>
        <h1>Traffic scene risk assessment</h1>
        <p className="hero-copy">
          Upload one road-scene image to inspect the model's risk label, explanation,
          and extracted features.
        </p>
        <p className="safety-note">
          Not for vehicle control, driving decisions, or other safety-critical use.
        </p>
      </header>

      <section className="workspace" aria-label="Image risk assessment">
        <div className="panel preview-panel">
          <div className="panel-heading">
            <div>
              <p className="step">Step 1</p>
              <h2>Choose a scene</h2>
            </div>
            {file && <span className="file-type">{file.type || "image"}</span>}
          </div>

          <div className={`preview ${previewUrl ? "has-image" : ""}`}>
            {previewUrl ? (
              <img src={previewUrl} alt={`Preview of ${file.name}`} />
            ) : (
              <div className="empty-preview" aria-hidden="true">
                <span className="empty-icon">ï¼‹</span>
                <span>Your image preview will appear here</span>
              </div>
            )}
          </div>

          <label className="file-button" htmlFor="file-input">
            Select JPEG, PNG, or WebP
          </label>
          <input
            className="visually-hidden"
            id="file-input"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileChange}
          />
          <p className="file-name" aria-live="polite">
            {file ? file.name : "No image selected"}
          </p>
        </div>

        <div className="panel analysis-panel">
          <div className="panel-heading">
            <div>
              <p className="step">Step 2</p>
              <h2>Run the assessment</h2>
            </div>
          </div>

          <form onSubmit={handleSubmit}>
            <p className="form-help">
              The service may return an unknown result when perception is incomplete or
              confidence is insufficient.
            </p>
            <div className="actions">
              <button className="primary-button" type="submit" disabled={!file || loading}>
                {loading ? "Analyzingâ€¦" : "Analyze image"}
              </button>
              {loading && (
                <button className="secondary-button" type="button" onClick={cancelAnalysis}>
                  Cancel
                </button>
              )}
            </div>
          </form>

          <div className="status-region" aria-live="polite" aria-atomic="true">
            {loading && (
              <div className="loading-state" role="status">
                <span className="spinner" aria-hidden="true" />
                Inspecting the traffic sceneâ€¦
              </div>
            )}

            {error && (
              <div className="error-state" role="alert">
                <strong>Assessment failed</strong>
                <span>{error}</span>
              </div>
            )}

            {result && (
              <article className={`result-card ${isUncertain ? "uncertain" : ""}`}>
                <div className="result-heading">
                  <div>
                    <p className="result-label">
                      {isUncertain ? "Insufficient confidence" : "Assessment result"}
                    </p>
                    <h3>{isUncertain ? "Review required" : "Scene analyzed"}</h3>
                  </div>
                  <span className={`risk-badge risk-${result.risk}`}>
                    {result.risk} risk
                  </span>
                </div>

                <dl className="result-summary">
                  <div>
                    <dt>Image ID</dt>
                    <dd>{result.image_id ?? "Not provided"}</dd>
                  </div>
                  <div>
                    <dt>Reason</dt>
                    <dd>{result.reason || "No explanation provided."}</dd>
                  </div>
                </dl>

                <DataTable caption="Extracted features" data={result.features} />
                <DataTable caption="Model versions" data={result.model_versions} />
              </article>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;

