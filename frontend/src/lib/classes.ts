export type AamiClass = "N" | "S" | "V" | "F" | "Q";

export const CLASS_COLORS: Record<string, string> = {
  N: "#94A3B8",
  S: "#E8A33D",
  V: "#A78BFA",
  F: "#4EA8DE",
  Q: "#6B7280",
};

export const CLASS_NAMES: Record<string, string> = {
  N: "Normal / bundle-branch",
  S: "Supraventricular ectopic",
  V: "Ventricular ectopic",
  F: "Fusion",
  Q: "Unknown / paced",
};

export const PARTITION_COLORS: Record<string, string> = {
  train: "#2BB3A3",
  test: "#E8A33D",
  excluded: "#5F6A7A",
};

export const PARTITION_LABELS: Record<string, string> = {
  train: "Training pool",
  test: "Held-out test",
  excluded: "Excluded (paced)",
};

export function classColor(c: string | null | undefined): string {
  return (c && CLASS_COLORS[c]) || "#5F6A7A";
}
