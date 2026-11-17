import { useEffect, useRef, useState } from "react";

import "./App.css";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
const API_URL = `${API_BASE_URL}/api/predict`;
const ALLOWED_RISKS = new Set(["low", "medium", "high", "unknown"]);
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const REQUEST_TIMEOUT_MS = 30_000;
const DETECTION_CATEGORIES = new Set([
  "person", "bicycle", "motorcycle", "car", "bus", "truck", "train", "traffic light", "traffic sign",
]);
const REASON_EXPLANATIONS = {
  hard_rule_triggered: "A conservative traffic rule identified a high-risk condition.",
  model_prediction: "The risk model produced this assessment from the detected scene features.",
  no_supported_detections: "No valid road users or traffic signals were detected. Try a clearer image with visible traffic objects.",
  no_rule_triggered: "Rules-only assessment: this image can produce high or unknown. No conservative high-risk rule matched, and statistical models are disabled.",
  feature_schema_error: "The detected scene could not be converted into the expected features. Try another image or check the service configuration.",
  perception_error: "The perception system could not process this image reliably. Try a clearer image or retry later.",
  invalid_features: "Image features could not be processed reliably. Try a clearer image or retry later.",
  invalid_model_output: "The risk model returned an unsupported result. Check the service configuration.",
  legacy_unspecified: "This service response does not include a specific reason code.",
};

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

  if (payload.detections !== undefined || payload.image_size !== undefined) {
    const size = payload.image_size;
    const detections = payload.detections;
    if (!Number.isInteger(size?.width) || size.width <= 0 || !Number.isInteger(size?.height) || size.height <= 0 || !Array.isArray(detections) || detections.length > 300) {
      throw new Error("The service returned invalid detection details. Please try again later.");
    }
    for (const detection of detections) {
      const box = detection?.bbox;
      if (!Number.isInteger(detection?.id) || detection.id < 0 || !DETECTION_CATEGORIES.has(detection?.category)
        || !Array.isArray(box) || box.length !== 4 || !box.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
        || box[0] > box[2] || box[1] > box[3]
        || !Number.isFinite(detection.confidence) || detection.confidence < 0 || detection.confidence > 1
        || !(detection.traffic_light_color === null || ["green", "red", "yellow", "unknown"].includes(detection.traffic_light_color))
        || (detection.category !== "traffic light" && detection.traffic_light_color !== null)
        || (detection.category === "traffic light" && detection.traffic_light_color === null)) {
        throw new Error("The service returned invalid detection details. Please try again later.");
      }
    }
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
  const [showDetections, setShowDetections] = useState(true);
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
    setShowDetections(true);
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

      const response = await fetch(`${API_URL}?include_detections=true`, {
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
  const reasonExplanation = result
    ? result.reason_code == null
      ? ""
      : (REASON_EXPLANATIONS[result.reason_code] ?? `The service returned an unrecognized reason code (${String(result.reason_code)}).`)
    : "";

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
              <div
                className="preview-stage"
                style={result?.image_size ? {
                  aspectRatio: `${result.image_size.width} / ${result.image_size.height}`,
                  "--preview-width-cap": `${320 * result.image_size.width / result.image_size.height}px`,
                  "--preview-mobile-width-cap": `${240 * result.image_size.width / result.image_size.height}px`,
                } : undefined}
              >
                <img src={previewUrl} alt={`Preview of ${file.name}`} />
                {showDetections && result?.image_size && result.detections?.length > 0 && (
                  <svg
                    className="detection-overlay"
                    viewBox={`0 0 ${result.image_size.width} ${result.image_size.height}`}
                    preserveAspectRatio="none"
                    role="img"
                    aria-label={`${result.detections.length} detected objects overlaid on the uploaded image`}
                  >
                    <title>Detected objects</title>
                    {result.detections.map((detection) => {
                      const [x1, y1, x2, y2] = detection.bbox;
                      const width = result.image_size.width;
                      const height = result.image_size.height;
                      const label = `${detection.category} ${(detection.confidence * 100).toFixed(0)}%`;
                      const labelAtRight = x1 * width > width - 220;
                      const labelX = labelAtRight ? width - 8 : Math.max(8, x1 * width);
                      const labelY = y1 * height < 40
                        ? Math.min(height - 4, y1 * height + 38)
                        : y1 * height - 5;
                      return (
                        <g key={detection.id}>
                          <title>{`${detection.category}, ${(detection.confidence * 100).toFixed(0)} percent confidence`}</title>
                          <rect x={x1 * width} y={y1 * height} width={(x2 - x1) * width} height={(y2 - y1) * height} />
                          <text x={labelX} y={labelY} textAnchor={labelAtRight ? "end" : "start"}>{label}</text>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>
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
          {result?.image_size && result.detections && (
            <label className="detection-toggle">
              <input type="checkbox" checked={showDetections} onChange={(event) => setShowDetections(event.target.checked)} />
              Show detected objects and confidence scores
            </label>
          )}
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
              The result includes a reason code and, when available, detections from the same inference pass.
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
                    <dd>
                      {reasonExplanation || result.reason || "No explanation provided."}
                      {reasonExplanation && result.reason && <><br /><span>{result.reason}</span></>}
                    </dd>
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

