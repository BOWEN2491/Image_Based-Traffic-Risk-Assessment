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
    expect(screen.getByText("central_tl_is_red")).toBeInTheDocument();
    expect(screen.getByText("risk_model")).toBeInTheDocument();
    expect(fetch).toHaveBeenCalledWith(
      "/api/predict",
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
        features: {},
        model_versions: {},
      }),
    );

    render(<App />);
    await selectImage(user);
    await user.click(screen.getByRole("button", { name: /analyze image/i }));

    expect(await screen.findByText("Review required")).toBeInTheDocument();
    expect(screen.getByText("unknown risk")).toBeInTheDocument();
    expect(screen.getByText("Insufficient confidence")).toBeInTheDocument();
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
});
