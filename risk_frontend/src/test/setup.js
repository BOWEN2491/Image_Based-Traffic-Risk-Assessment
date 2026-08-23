import "@testing-library/jest-dom/vitest";
import { vi } from "vitest";

Object.defineProperty(URL, "createObjectURL", {
  configurable: true,
  value: vi.fn(() => "blob:preview"),
});

Object.defineProperty(URL, "revokeObjectURL", {
  configurable: true,
  value: vi.fn(),
});
