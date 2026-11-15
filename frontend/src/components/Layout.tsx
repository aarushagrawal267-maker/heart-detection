import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Activity,
  BookOpen,
  BrainCircuit,
  Database,
  FlaskConical,
  Gauge,
  History,
  LayoutDashboard,
  Menu,
  ScanLine,
  Waves,
  X,
} from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";
import { api } from "../api/client";
import { num } from "../lib/format";

const NAV = [
  {
    group: "Signals",
    items: [
      { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
      { to: "/explorer", label: "Signal Explorer", icon: Waves },
      { to: "/classify", label: "Beat Classification", icon: ScanLine },
      { to: "/analysis", label: "Full ECG Analysis", icon: Activity },
    ],
  },
  { group: "Data", items: [{ to: "/dataset", label: "Dataset Explorer", icon: Database }] },
  {
    group: "Models",
    items: [
      { to: "/lab", label: "Model Laboratory", icon: FlaskConical },
      { to: "/evaluation", label: "Model Evaluation", icon: Gauge },
      { to: "/experiments", label: "Experiment History", icon: History },
    ],
  },
  { group: "Reference", items: [{ to: "/research", label: "Research & Docs", icon: BookOpen }] },
];

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden>
        <rect width="32" height="32" rx="8" fill="#10151C" stroke="#212A36" />
        <path d="M3 18h7l2-4 3 10 4-17 3 11h7" fill="none" stroke="#F2555A" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="leading-tight">
        <div className="text-[14px] font-semibold tracking-tight text-fg">QRS Atlas</div>
        <div className="text-2xs text-fg-dim">ECG signal intelligence</div>
      </div>
    </div>
  );
}

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex h-full flex-col gap-6 overflow-y-auto px-3 py-5">
      <div className="px-2">
        <Logo />
      </div>
      {NAV.map((g) => (
        <div key={g.group}>
          <div className="label mb-1.5 px-2.5">{g.group}</div>
          <ul className="space-y-0.5">
            {g.items.map((it) => (
              <li key={it.to}>
                <NavLink
                  to={it.to}
                  end={it.end}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    clsx(
                      "group flex items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] transition-colors",
                      isActive ? "bg-ink-700 text-fg" : "text-fg-muted hover:bg-ink-750 hover:text-fg",
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      <it.icon className={clsx("h-4 w-4", isActive ? "text-teal" : "text-fg-dim group-hover:text-fg-muted")} />
                      {it.label}
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
      <div className="mt-auto rounded-lg border border-line bg-ink-850 p-3">
        <div className="flex items-center gap-1.5 text-2xs font-semibold uppercase tracking-[0.08em] text-teal">
          <BrainCircuit className="h-3.5 w-3.5" /> Research use only
        </div>
        <p className="mt-1.5 text-2xs leading-relaxed text-fg-dim">
          Not a medical device. Model outputs are research results, not diagnoses.
        </p>
      </div>
    </nav>
  );
}

function StatusBar() {
  const ds = useQuery({ queryKey: ["dataset-status"], queryFn: api.datasetStatus, refetchInterval: 15000 });
  const model = useQuery({ queryKey: ["active-model"], queryFn: api.activeModel, refetchInterval: 15000 });
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000 });
  const chip = (ok: boolean | undefined, label: string, value: string) => (
    <span className="inline-flex items-center gap-1.5 rounded-md border border-line bg-ink-800 px-2 py-1 text-2xs">
      <span className={clsx("h-1.5 w-1.5 rounded-full", ok === undefined ? "bg-fg-dim" : ok ? "bg-teal" : "bg-trace")} />
      <span className="text-fg-dim">{label}</span>
      <span className="num text-fg-muted">{value}</span>
    </span>
  );
  return (
    <div className="hidden items-center gap-2 md:flex" data-testid="status-bar">
      {chip(health.isSuccess ? true : health.isError ? false : undefined, "API", health.isSuccess ? health.data.device.toUpperCase() : health.isError ? "offline" : "…")}
      {chip(
        ds.data ? ds.data.present && ds.data.processed : undefined,
        "MIT-BIH",
        ds.data ? (ds.data.present ? `${ds.data.n_records_expected} rec · ${ds.data.n_subjects} subj` : "not downloaded") : "…",
      )}
      {chip(
        model.isSuccess ? !!model.data : undefined,
        "Model",
        model.data ? `ECGNet · val F1 ${num(model.data.summary.val_macro_f1 ?? null, 2)}` : model.isSuccess ? "none trained" : "…",
      )}
    </div>
  );
}

export default function Layout() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  return (
    <div className="flex h-full">
      <aside className="hidden w-60 shrink-0 border-r border-line bg-ink-850 lg:block">
        <Sidebar />
      </aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-64 border-r border-line bg-ink-850">
            <Sidebar onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-line bg-ink-900/80 px-4 backdrop-blur">
          <div className="flex items-center gap-2">
            <button className="btn px-2 lg:hidden" onClick={() => setOpen(true)} aria-label="Open navigation">
              {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </button>
            <span className="text-2xs text-fg-dim">
              MIT-BIH Arrhythmia Database · patient-independent evaluation
            </span>
          </div>
          <StatusBar />
        </header>
        <main className="min-w-0 flex-1 overflow-y-auto" key={loc.pathname}>
          <div className="mx-auto max-w-[1440px] px-4 py-6 sm:px-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
