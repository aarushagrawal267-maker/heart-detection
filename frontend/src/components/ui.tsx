import clsx from "clsx";
import { AlertTriangle, Info, Loader2, ShieldAlert } from "lucide-react";
import type { ReactNode } from "react";
import { CLASS_NAMES, classColor, PARTITION_COLORS, PARTITION_LABELS } from "../lib/classes";

export function Panel({
  title,
  sub,
  actions,
  children,
  className,
  bodyClass,
  testId,
}: {
  title?: ReactNode;
  sub?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
  testId?: string;
}) {
  return (
    <section className={clsx("panel min-w-0", className)} data-testid={testId}>
      {(title || actions) && (
        <header className="panel-head">
          <div className="min-w-0">
            {title && <h2 className="panel-title truncate">{title}</h2>}
            {sub && <p className="panel-sub mt-0.5">{sub}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={clsx("p-4", bodyClass)}>{children}</div>
    </section>
  );
}

export function Stat({
  label,
  value,
  sub,
  accent,
  testId,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: string;
  testId?: string;
}) {
  return (
    <div className="panel px-4 py-3.5" data-testid={testId}>
      <div className="label">{label}</div>
      <div className="num mt-1.5 text-2xl font-medium tracking-tight" style={accent ? { color: accent } : undefined}>
        {value}
      </div>
      {sub && <div className="mt-1 text-2xs text-fg-dim">{sub}</div>}
    </div>
  );
}

export function ClassChip({ cls, label, size = "sm" }: { cls: string | null | undefined; label?: string; size?: "sm" | "lg" }) {
  const c = classColor(cls);
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-md border font-mono font-medium",
        size === "lg" ? "px-2.5 py-1 text-sm" : "px-1.5 py-0.5 text-2xs",
      )}
      style={{ color: c, borderColor: c + "55", background: c + "14" }}
      title={cls ? CLASS_NAMES[cls] : undefined}
    >
      <span className="inline-block h-1.5 w-1.5 rounded-full" style={{ background: c }} />
      {label ?? cls ?? "—"}
    </span>
  );
}

export function PartitionChip({ partition }: { partition: string }) {
  const c = PARTITION_COLORS[partition] ?? "#5F6A7A";
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-md border px-1.5 py-0.5 text-2xs font-medium"
      style={{ color: c, borderColor: c + "44", background: c + "12" }}
    >
      {PARTITION_LABELS[partition] ?? partition}
    </span>
  );
}

export function StatusDot({ status }: { status: string }) {
  const color =
    status === "completed"
      ? "#2BB3A3"
      : status === "running" || status === "cancelling" || status === "queued"
        ? "#E8A33D"
        : status === "failed"
          ? "#F2555A"
          : "#5F6A7A";
  return (
    <span className="inline-flex items-center gap-1.5 text-2xs font-medium capitalize" style={{ color }}>
      <span
        className={clsx("h-1.5 w-1.5 rounded-full", (status === "running" || status === "queued") && "animate-pulse")}
        style={{ background: color }}
      />
      {status}
    </span>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-8 text-xs text-fg-dim" role="status">
      <Loader2 className="h-4 w-4 animate-spin text-teal" />
      {label}…
    </div>
  );
}

export function ErrorState({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="flex items-start gap-2 rounded-lg border border-trace/30 bg-trace/5 px-3 py-2.5 text-xs text-trace">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{msg}</span>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line-strong px-6 py-10 text-center">
      <div className="text-sm font-medium text-fg">{title}</div>
      {children && <div className="max-w-md text-xs text-fg-dim">{children}</div>}
    </div>
  );
}

export function Note({ children, tone = "info" }: { children: ReactNode; tone?: "info" | "warn" | "safety" }) {
  const styles = {
    info: "border-line-strong bg-ink-750 text-fg-muted",
    warn: "border-cls-S/30 bg-cls-S/5 text-cls-S",
    safety: "border-teal/25 bg-teal/5 text-fg-muted",
  }[tone];
  const Icon = tone === "safety" ? ShieldAlert : tone === "warn" ? AlertTriangle : Info;
  return (
    <div className={clsx("flex items-start gap-2 rounded-lg border px-3 py-2 text-2xs leading-relaxed", styles)}>
      <Icon className={clsx("mt-px h-3.5 w-3.5 shrink-0", tone === "safety" && "text-teal")} />
      <div>{children}</div>
    </div>
  );
}

export function ResearchNotice() {
  return (
    <Note tone="safety">
      <strong className="font-semibold text-fg">Research-model output.</strong> For education and research only — not a
      medical device, not a diagnosis, and not a substitute for a clinician.
    </Note>
  );
}

export function PageHeader({ title, sub, actions }: { title: string; sub?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-fg">{title}</h1>
        {sub && <p className="mt-1 max-w-3xl text-xs leading-relaxed text-fg-muted">{sub}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Seg<T extends string | number>({
  value,
  options,
  onChange,
  testId,
}: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  testId?: string;
}) {
  return (
    <div className="seg" data-testid={testId}>
      {options.map((o) => (
        <button key={String(o.value)} data-active={o.value === value} onClick={() => onChange(o.value)} type="button">
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="block">
      <span className="label">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-1 block text-2xs text-fg-dim">{hint}</span>}
    </label>
  );
}

export function Bar({ value, color, max = 1 }: { value: number; color: string; max?: number }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-600">
      <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(1, value / max)) * 100}%`, background: color }} />
    </div>
  );
}
