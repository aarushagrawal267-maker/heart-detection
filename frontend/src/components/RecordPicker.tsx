import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { PARTITION_LABELS } from "../lib/classes";

export default function RecordPicker({
  value,
  onChange,
  testId = "record-picker",
}: {
  value: string;
  onChange: (r: string) => void;
  testId?: string;
}) {
  const q = useQuery({ queryKey: ["records"], queryFn: api.records, staleTime: Infinity });
  const groups: Record<string, typeof q.data> = {};
  for (const r of q.data ?? []) (groups[r.partition] ??= []).push(r);
  return (
    <select className="input w-auto min-w-[170px]" value={value} onChange={(e) => onChange(e.target.value)} data-testid={testId}>
      {(["train", "test", "excluded"] as const).map((p) =>
        groups[p]?.length ? (
          <optgroup key={p} label={PARTITION_LABELS[p]}>
            {groups[p]!.map((r) => (
              <option key={r.record} value={r.record}>
                Record {r.record} · {r.analysis_lead}
                {r.paced ? " · paced" : ""}
              </option>
            ))}
          </optgroup>
        ) : null,
      )}
    </select>
  );
}
