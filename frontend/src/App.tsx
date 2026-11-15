import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Analysis from "./pages/Analysis";
import Classify from "./pages/Classify";
import Dataset from "./pages/Dataset";
import Docs from "./pages/Docs";
import Evaluation from "./pages/Evaluation";
import Experiments from "./pages/Experiments";
import Explorer from "./pages/Explorer";
import Lab from "./pages/Lab";
import Overview from "./pages/Overview";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="explorer" element={<Explorer />} />
        <Route path="classify" element={<Classify />} />
        <Route path="analysis" element={<Analysis />} />
        <Route path="dataset" element={<Dataset />} />
        <Route path="lab" element={<Lab />} />
        <Route path="evaluation" element={<Evaluation />} />
        <Route path="experiments" element={<Experiments />} />
        <Route path="docs" element={<Docs />} />
        <Route path="*" element={<Overview />} />
      </Route>
    </Routes>
  );
}
