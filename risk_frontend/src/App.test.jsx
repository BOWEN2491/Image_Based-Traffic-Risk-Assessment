import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": "application/json" }),
    json: vi.fn().mockResolvedValue(payload),
  };
}

async function selectImage(user) {
  const image = new File(["image data"], "intersection.jpg", {
    type: "image/jpeg",
  });
  await user.upload(screen.getByLabelText(/select jpeg/i), image);
  return image;
}

describe("risk assessment", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("renders a successful assessment with features and model versions", async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValue(
      jsonResponse({
        status: "ok",
        risk: "high",
        image_id: "img-123",
        reason: "Central traffic light is red.",
        reason_code: "hard_rule_triggered",
        features: { central_tl_is_red: true, nearest_vehicle_distance: 8.4 },
        model_versions: { risk_model: "0.1.0" },
      }),
    );

    const { container } = render(<App />);
    const image = await selectImage(user);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));

    expect(await screen.findByText("Scene analyzed")).toBeInTheDocument();
    expect(screen.getByText("high risk")).toBeInTheDocument();
    expect(screen.getByText("Central traffic light is red.")).toBeInTheDocument();
    expect(screen.getByText(/conservative traffic rule identified a high-risk condition/i)).toBeInTheDocument();
    expect(screen.getByText("central_tl_is_red")).toBeInTheDocument();
    expect(screen.getByText("risk_model")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      "/api/predict?include_detections=true",
      expect.objectContaining({ method: "POST", signal: expect.any(AbortSignal) }),
    );
    expect(fetch.mock.calls[0][1].body.get("file")).toBe(image);
    expect(container.querySelector("[role='alert']")).not.toBeInTheDocument();
  });

  it("presents uncertain and unknown as a review-required state", async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValue(
      jsonResponse({
        status: "uncertain",
        risk: "unknown",
        image_id: "img-unknown",
        reason: "No reliable detections were produced.",
        reason_code: "no_supported_detections",


        features: {},
        model_versions: {},
      }),
    );

    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));

    expect(await screen.findByText("Review required")).toBeInTheDocument();
    expect(screen.getByText("unknown risk")).toBeInTheDocument();
    expect(screen.getByText(/no valid road users or traffic signals were detected/i)).toBeInTheDocument();
  });

  it("renders optional same-image detections and lets the user hide their overlay", async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValue(jsonResponse({
      status: "uncertain", risk: "unknown", image_id: "img-detections",
      reason: "No conservative rule triggered", reason_code: "no_rule_triggered",
      features: {}, model_versions: {}, image_size: { width: 100, height: 50 },
      detections: [{ id: 4, category: "traffic light", bbox: [0.1, 0.2, 0.4, 0.6], confidence: 0.91, traffic_light_color: "unknown" }],
    }));
    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));

    expect(await screen.findByRole("img", { name: /1 detected object/i })).toBeInTheDocument();
    expect(screen.getByText(/rules-only assessment/i)).toBeInTheDocument();
    await user.click(screen.getByRole("checkbox", { name: /show detected objects/i }));
    expect(screen.queryByRole("img", { name: /detected objects overlaid/i })).not.toBeInTheDocument();
  });

  it("rejects malformed detection boxes and safely handles unknown reason codes", async () => {
    const user = userEvent.setup();
    fetch.mockResolvedValue(jsonResponse({
      status: "ok", risk: "low", image_id: "bad-box", reason: "Legacy detail",
      reason_code: "future_reason_code", features: {}, model_versions: {},
      image_size: { width: 100, height: 50 },
      detections: [{ id: 0, category: "car", bbox: [0.8, 0.2, 0.3, 0.6], confidence: 0.5, traffic_light_color: null }],
    }));
    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/invalid detection details/i);
    expect(screen.queryByText(/unrecognized reason code/i)).not.toBeInTheDocument();
  });

  it.each([
    [422, { detail: { code: "invalid_image", message: "Image dimensions are invalid" } }, "Image dimensions are invalid"],
    [503, { detail: { code: "service_busy", message: "Inference queue is full" } }, "Inference queue is full"],
  ])("shows a structured %s API error", async (status, payload, message) => {
    const user = userEvent.setup();
    fetch.mockResolvedValue(jsonResponse(payload, status));

    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(screen.queryByText("Scene analyzed")).not.toBeInTheDocument();
  });

  it.each([
    ["ok", "unknown"],
    ["uncertain", "low"],
  ])("rejects the invalid status/risk pair %s/%s", async (status, risk) => {
    const user = userEvent.setup();
    fetch.mockResolvedValue(
      jsonResponse({ status, risk, image_id: "bad-pair", reason: "invalid contract" }),
    );

    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The service returned an unexpected response",
    );
    expect(screen.queryByText("Scene analyzed")).not.toBeInTheDocument();
    expect(screen.queryByText("Review required")).not.toBeInTheDocument();
  });

  it("prevents duplicate submissions while a request is in flight", async () => {
    const user = userEvent.setup();
    let resolveRequest;
    fetch.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRequest = resolve;
        }),
    );

    render(<App />);
    await selectImage(user);
    const submit = screen.getByRole("button", { name: /analyze image/i });
    const form = submit.closest("form");

    fireEvent.submit(form);
    fireEvent.submit(form);

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: /analyzing/i })).toBeDisabled();

    resolveRequest(
      jsonResponse({
        status: "ok",
        risk: "low",
        image_id: "img-456",
        reason: "No elevated rule matched.",
        features: {},
        model_versions: {},
      }),
    );

    await waitFor(() => expect(screen.getByText("Scene analyzed")).toBeInTheDocument());
  });

  it("cancels an in-flight request and clears any stale overlay", async () => {
    const user = userEvent.setup();
    fetch.mockImplementation((_url, { signal }) => new Promise((_resolve, reject) => {
      signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    }));

    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));
    await user.click(screen.getByRole("button", { name: /cancel/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Analysis canceled.");
    expect(screen.queryByRole("img", { name: /detected objects overlaid/i })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /analyze image/i })).toBeEnabled();
  });
});
