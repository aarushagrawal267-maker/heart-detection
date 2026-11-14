import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ConfusionMatrix, ScoreBars } from "./metrics";
import { ClassChip, ResearchNotice, Seg } from "./ui";
import { CLASS_COLORS, classColor } from "../lib/classes";
import type { Metrics } from "../api/client";

describe("UI primitives", () => {
  it("always shows the research-use notice wording", () => {
    render(<ResearchNotice />);
    expect(screen.getByText(/Research-model output/)).toBeInTheDocument();
    expect(screen.getByText(/not a diagnosis/)).toBeInTheDocument();
  });

  it("colours class chips consistently", () => {
    render(<ClassChip cls="V" />);
    expect(screen.getByText("V")).toHaveStyle({ color: CLASS_COLORS.V });
    expect(classColor("unknown")).toBe("#5F6A7A");
  });

  it("segmented control reports changes", () => {
    const onChange = vi.fn();
    render(<Seg value="a" onChange={onChange} options={[{ value: "a", label: "A" }, { value: "b", label: "B" }]} />);
    fireEvent.click(screen.getByText("B"));
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("renders class scores", () => {
    render(<ScoreBars scores={{ N: 0.9, S: 0.05, V: 0.04, F: 0.01 }} highlight="N" />);
    expect(screen.getByText("90.0%")).toBeInTheDocument();
  });
});

describe("ConfusionMatrix", () => {
  const metrics: Metrics = {
    classes: ["N", "S"],
    n: 100,
    accuracy: 0.91,
    macro_f1: 0.75,
    weighted_f1: 0.9,
    macro_recall: 0.78,
    confusion_matrix: [
      [85, 5],
      [4, 6],
    ],
    per_class: {},
  };
  it("shows row percentages and toggles to counts", () => {
    render(<ConfusionMatrix metrics={metrics} />);
    expect(screen.getByText("94%")).toBeInTheDocument();
    expect(screen.getByText("60%")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Counts"));
    expect(screen.getByText("85")).toBeInTheDocument();
  });
});
