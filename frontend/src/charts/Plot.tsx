import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export const PLOT_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "Inter, system-ui, sans-serif", size: 11, color: "#9AA3B2" },
  margin: { l: 48, r: 12, t: 12, b: 40 },
  xaxis: { gridcolor: "#1A212B", zerolinecolor: "#273241", linecolor: "#273241", tickfont: { family: "JetBrains Mono, monospace", size: 11 } },
  yaxis: { gridcolor: "#1A212B", zerolinecolor: "#273241", linecolor: "#273241", tickfont: { family: "JetBrains Mono, monospace", size: 11 } },
  legend: { orientation: "h", y: -0.22, font: { size: 11 } },
  hoverlabel: { bgcolor: "#131922", bordercolor: "#2D3847", font: { family: "JetBrains Mono, monospace", size: 11, color: "#E4E8EE" } },
};

interface Props {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  height?: number;
  testId?: string;
}

function merge(layout: Partial<Plotly.Layout> = {}): Partial<Plotly.Layout> {
  return {
    ...PLOT_LAYOUT,
    ...layout,
    xaxis: { ...PLOT_LAYOUT.xaxis, ...(layout.xaxis ?? {}) },
    yaxis: { ...PLOT_LAYOUT.yaxis, ...(layout.yaxis ?? {}) },
    margin: { ...PLOT_LAYOUT.margin, ...(layout.margin ?? {}) },
  };
}

/** Thin Plotly wrapper with the platform's dark scientific theme. */
export default function Plot({ data, layout, height = 300, testId }: Props) {
  const el = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!el.current) return;
    Plotly.react(el.current, data, { ...merge(layout), height, autosize: true }, {
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
    });
  }, [data, layout, height]);
  useEffect(() => {
    const node = el.current;
    return () => {
      if (node) Plotly.purge(node);
    };
  }, []);
  return <div ref={el} style={{ height }} className="w-full" data-testid={testId} />;
}
