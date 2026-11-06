export function relativeTime(iso: string): string {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return new Date(iso).toLocaleDateString();
}

export function tokens(n: number | null | undefined): string {
  if (n == null) return "–";
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
}

export function seconds(s: number | null | undefined): string {
  if (s == null) return "";
  return s < 1 ? `${Math.round(s * 1000)}ms` : `${s.toFixed(1)}s`;
}

export function prettyArgs(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

export function inlineArgs(value: unknown, limit = 90): string {
  const text = typeof value === "string" ? value : JSON.stringify(value ?? {});
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}
