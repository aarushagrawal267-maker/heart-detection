import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests run against the real backend and frontend
 * (start them first with start.ps1, or `uvicorn` + `vite`).
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: process.env.QRS_FRONTEND_URL ?? "http://127.0.0.1:5173",
    trace: "off",
    viewport: { width: 1440, height: 1000 },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } } }],
});
