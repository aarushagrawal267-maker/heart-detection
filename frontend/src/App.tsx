import { lazy, Suspense, type ReactNode } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";

// Pages are split into separate chunks so the first load stays small; the large Plotly
// bundle is only fetched by pages that draw analytical charts.
const Overview = lazy(() => import("./pages/Overview"));
const Explorer = lazy(() => import("./pages/Explorer"));
const Classify = lazy(() => import("./pages/Classify"));
const Analysis = lazy(() => import("./pages/Analysis"));
const Dataset = lazy(() => import("./pages/Dataset"));
const Lab = lazy(() => import("./pages/Lab"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Experiments = lazy(() => import("./pages/Experiments"));
const Docs = lazy(() => import("./pages/Docs"));

function PageFallback() {
  return (
    <div className="flex h-64 items-center justify-center text-sm text-slate-400" role="status" aria-live="polite">
      Loading page…
    </div>
  );
}

function page(node: ReactNode) {
  return <Suspense fallback={<PageFallback />}>{node}</Suspense>;
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={page(<Overview />)} />
        <Route path="explorer" element={page(<Explorer />)} />
        <Route path="classify" element={page(<Classify />)} />
        <Route path="analysis" element={page(<Analysis />)} />
        <Route path="dataset" element={page(<Dataset />)} />
        <Route path="lab" element={page(<Lab />)} />
        <Route path="evaluation" element={page(<Evaluation />)} />
        <Route path="experiments" element={page(<Experiments />)} />
        <Route path="research" element={page(<Docs />)} />
        <Route path="docs" element={page(<Docs />)} />
        <Route path="*" element={page(<Overview />)} />
      </Route>
    </Routes>
  );
}
