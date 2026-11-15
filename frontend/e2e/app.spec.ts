import { expect, test } from "@playwright/test";

test.describe("QRS Atlas end-to-end (real data)", () => {
  test("overview shows real dataset figures and the research notice", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "ECG Intelligence Overview" })).toBeVisible();
    const stats = page.getByTestId("overview-stats");
    await expect(stats).toContainText("48");
    await expect(stats).toContainText("109,444");
    await expect(page.getByText("Research-model output.").first()).toBeVisible();
    await expect(page.getByTestId("hero-ecg").locator("canvas")).toBeVisible();
    await expect(page.getByTestId("status-bar")).toContainText("48 rec · 47 subj");
  });

  test("signal explorer renders a waveform and selects a beat", async ({ page }) => {
    await page.goto("/explorer?record=100");
    const chart = page.getByTestId("explorer-ecg");
    await expect(chart.locator("canvas")).toBeVisible();
    await expect(page.getByTestId("minimap")).toBeVisible();
    const over = chart.locator(".u-over");
    const box = await over.boundingBox();
    await over.click({ position: { x: box!.width / 2, y: box!.height / 2 } });
    await expect(page.getByTestId("beat-panel")).toContainText("Beat #");
    await expect(page.getByTestId("beat-details")).toContainText("Reference class");
    // panning must move the displayed window (regression: stale x-range)
    await page.getByRole("button", { name: "Pan right" }).click();
    await expect(page.getByText(/0:08\.0 – 0:18\.0/)).toBeVisible();
    const xs = await page.evaluate(() => {
      const labels = Array.from(document.querySelectorAll('[data-testid="explorer-ecg"] .u-axis')[0]?.querySelectorAll(".u-label, div") ?? []);
      return labels.map((d) => d.textContent ?? "").join(" ");
    });
    expect(xs).not.toContain("0:00.0");
    await page.getByTestId("record-picker").selectOption("114");
    await expect(page.getByTestId("channel-picker")).toHaveValue("MLII");
    await page.getByTestId("signal-mode").getByText("Preprocessed").click();
    await expect(page.getByText("baseline-corrected, 45 Hz low-pass")).toBeVisible();
  });

  test("classification centre runs the trained CNN and compares with the reference", async ({ page }) => {
    await page.goto("/classify?record=208");
    await page.getByTestId("beat-selector").getByRole("button", { name: /^V/ }).click();
    await page.getByTestId("run-classify").click();
    const result = page.getByTestId("prediction-result");
    await expect(result).toContainText("Expert reference");
    await expect(result).toContainText("CNN prediction");
    await expect(page.getByTestId("agreement")).toBeVisible();
    await expect(page.getByTestId("score-bars")).toContainText("%");
    await expect(page.getByTestId("classified-beat").locator("canvas")).toBeVisible();
    await page.getByTestId("explain-method").getByText("Grad-CAM").click();
    await expect(page.getByText(/1D Grad-CAM/)).toBeVisible();
    await expect(page.getByText(/not a diagnosis/i).first()).toBeVisible();
  });

  test("full ECG analysis classifies every beat of a recording", async ({ page }) => {
    await page.goto("/analysis?record=100");
    await page.getByTestId("run-analysis").click();
    await expect(page.getByTestId("analysis-stats")).toContainText("2,271", { timeout: 60_000 });
    await expect(page.getByTestId("analysis-distribution")).toContainText("Reference");
    await expect(page.getByTestId("timeline")).toBeVisible();
    await page.getByTestId("analysis-mode").getByText("Automatic R-peak detection").click();
    await page.getByTestId("run-analysis").click();
    await expect(page.getByTestId("detection-stats")).toContainText("Detector Se", { timeout: 60_000 });
  });

  test("dataset explorer lists 48 recordings and passes all leakage checks", async ({ page }) => {
    await page.goto("/dataset");
    const rows = page.getByTestId("inventory").locator("tbody tr");
    await expect(rows).toHaveCount(48, { timeout: 90_000 });
    const split = page.getByTestId("split-panel");
    await expect(split).toContainText("Records 201 and 202 share one partition");
    await expect(split).toContainText("Record 202 moved from DS2");
    await expect(page.getByTestId("provenance")).toContainText("Open Data Commons");
  });

  test("model laboratory shows the architecture and live training state", async ({ page, request }) => {
    await page.goto("/lab");
    await expect(page.getByTestId("architecture")).toContainText("Conv1d(k=7)");
    await expect(page.getByTestId("start-training")).toBeVisible();
    const current = await (await request.get("/api/training/current")).json();
    if (current) {
      // a job is already running (e.g. model selection): the monitor must show real progress
      await expect(page.getByTestId("training-monitor")).toContainText(current.name);
      return;
    }
    await page.getByTestId("exp-name").fill("e2e smoke (cancelled)");
    await page.getByTestId("cfg-epochs").fill("1");
    await page.getByTestId("start-training").click();
    await expect(page.getByTestId("training-message")).toContainText(/epoch|loading|starting|evaluating|data/, { timeout: 60_000 });
    await page.getByTestId("cancel-training").click();
    await expect(page.getByTestId("training-monitor")).toContainText("cancelled", { timeout: 90_000 });
    const exps = await (await request.get("/api/experiments")).json();
    for (const e of exps.filter((x: { name: string }) => x.name === "e2e smoke (cancelled)")) {
      await request.delete(`/api/experiments/${e.id}`);
    }
  });

  test("evaluation page shows confusion matrix, per-class metrics and the exact split", async ({ page }) => {
    await page.goto("/evaluation");
    await expect(page.getByTestId("eval-confusion")).toBeVisible();
    await expect(page.getByTestId("per-class-table")).toContainText("Sensitivity");
    await expect(page.getByTestId("model-comparison")).toContainText("Majority-class baseline");
    await expect(page.getByTestId("exact-split")).toContainText("Held-out test");
  });

  test("experiment history lists runs and compares two of them", async ({ page }) => {
    await page.goto("/experiments");
    const table = page.getByTestId("experiment-table");
    await expect(table).toContainText("Random forest");
    const boxes = table.locator("tbody input[type=checkbox]");
    await boxes.nth(0).check();
    await boxes.nth(1).check();
    await expect(page.getByTestId("comparison")).toContainText("validation macro F1");
  });

  test("documentation covers methodology and limitations", async ({ page }) => {
    await page.goto("/docs");
    const docs = page.getByTestId("docs");
    await expect(docs).toContainText("Patient-independent evaluation");
    await expect(docs).toContainText("record 202 is moved to the training side", { ignoreCase: true });
    await expect(docs).toContainText("Not a medical device");
  });
});
