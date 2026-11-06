import { useState } from "react";
import type { RunEvent } from "../types";
import { inlineArgs, prettyArgs, seconds, tokens } from "../format";

/**
 * Turns a run's flat event stream into timeline items, then renders each by kind.
 *
 * To support a new event type (e.g. subagent.started, approval.required), add a case to
 * `buildTimeline` and a renderer to `ItemView`; anything unrecognised still shows up via the
 * generic "other" item, so new backend events are never silently hidden.
 */

type ToolItem = {
  kind: "tool";
  key: string;
  step: number;
  callId: string;
  name: string;
  args: unknown;
  status: "running" | "ok" | "error";
  content?: string;
  chars?: number;
  errorCode?: string | null;
  duration?: number;
  offload?: { artifact_id: string; tokens?: number };
};

type Item =
  | { kind: "step"; key: string; step: number }
  | { kind: "thought"; key: string; step: number; text: string }
  | ToolItem
  | { kind: "compacted"; key: string; step: number; data: RunEvent["data"] }
  | { kind: "reduced"; key: string; step: number; data: RunEvent["data"] }
  | { kind: "other"; key: string; step: number; event: RunEvent };

// Lifecycle events shown elsewhere (the run header/footer), not as timeline rows.
const HIDDEN = new Set(["run.started", "run.finished", "run.persisted", "run.persist_failed"]);

export function buildTimeline(events: RunEvent[]): Item[] {
  const items: Item[] = [];
  const tools = new Map<string, ToolItem>();
  for (const e of events) {
    const key = `e${e.seq}`;
    switch (e.type) {
      case "step.started":
        items.push({ kind: "step", key, step: e.step });
        break;
      case "llm.response":
        // Text that comes with tool calls is the model thinking out loud; text without tool
        // calls is the final answer, which the run view shows on its own.
        if (e.data.text && e.data.tool_calls?.length) items.push({ kind: "thought", key, step: e.step, text: e.data.text });
        break;
      case "tool.started": {
        const item: ToolItem = {
          kind: "tool", key, step: e.step, callId: e.data.call_id, name: e.data.name, args: e.data.arguments,
          status: "running",
        };
        tools.set(item.callId, item);
        items.push(item);
        break;
      }
      case "tool.completed": {
        const item = tools.get(e.data.call_id);
        if (item) {
          item.status = e.data.is_error ? "error" : "ok";
          item.content = e.data.content;
          item.chars = e.data.chars;
          item.errorCode = e.data.error_code;
          item.duration = e.data.duration;
        }
        break;
      }
      case "tool.offloaded": {
        const item = tools.get(e.data.call_id);
        if (item) item.offload = { artifact_id: e.data.artifact_id, tokens: e.data.tokens };
        break;
      }
      case "context.compacted":
        items.push({ kind: "compacted", key, step: e.step, data: e.data });
        break;
      case "context.reduced":
        items.push({ kind: "reduced", key, step: e.step, data: e.data });
        break;
      default:
        if (!HIDDEN.has(e.type)) items.push({ kind: "other", key, step: e.step, event: e });
    }
  }
  return items;
}

export function Timeline({ events, onOpenArtifact }: { events: RunEvent[]; onOpenArtifact: (id: string) => void }) {
  const items = buildTimeline(events);
  if (!items.length) return <div className="timeline-empty">Waiting for the model…</div>;
  return (
    <ol className="timeline">
      {items.map((item) => (
        <ItemView key={item.key} item={item} onOpenArtifact={onOpenArtifact} />
      ))}
    </ol>
  );
}

function ItemView({ item, onOpenArtifact }: { item: Item; onOpenArtifact: (id: string) => void }) {
  switch (item.kind) {
    case "step":
      return <li className="tl-step">Step {item.step}</li>;
    case "thought":
      return <li className="tl-thought">{item.text}</li>;
    case "tool":
      return <ToolCard item={item} onOpenArtifact={onOpenArtifact} />;
    case "compacted": {
      const d = item.data;
      return (
        <li className="tl-notice tl-compacted">
          <strong>Memory compacted</strong>
          {d.reason === "provider_overflow" ? " (the provider said the request was too long)" : ""}: messages{" "}
          {d.messages?.[0]}–{d.messages?.[1]} (~{tokens(d.region_tokens)} tokens) were replaced by a ~
          {tokens(d.summary_tokens)}-token summary.
          {d.archive && (
            <>
              {" "}Originals archived as{" "}
              <button className="link" onClick={() => onOpenArtifact(d.archive)}>
                {d.archive}
              </button>
              .
            </>
          )}
        </li>
      );
    }
    case "reduced":
      return (
        <li className="tl-notice tl-reduced">
          <strong>Context trimmed</strong>: dropped the {item.data.groups_dropped} oldest tool exchange(s) from what
          the model sees, to fit the budget (~{tokens(item.data.estimated_tokens)} tokens sent).
        </li>
      );
    case "other":
      return (
        <li className="tl-notice">
          <strong>{item.event.type}</strong> <code>{inlineArgs(item.event.data, 160)}</code>
        </li>
      );
  }
}

function ToolCard({ item, onOpenArtifact }: { item: ToolItem; onOpenArtifact: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <li className={`tl-tool status-${item.status}`}>
      <button className="tool-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="tool-dot" aria-hidden />
        <span className="tool-name">{item.name}</span>
        <code className="tool-args">{inlineArgs(item.args)}</code>
        <span className="tool-meta">
          {item.status === "running" ? "running…" : item.status === "error" ? (item.errorCode ?? "error") : `${item.chars?.toLocaleString()} chars`}
          {item.duration != null && ` · ${seconds(item.duration)}`}
        </span>
        {item.offload && <span className="badge badge-art">artifact</span>}
      </button>
      {open && (
        <div className="tool-body">
          <div className="label">Arguments</div>
          <pre>{prettyArgs(item.args)}</pre>
          {item.offload && (
            <p className="tool-offload">
              Too large to keep in the conversation (~{tokens(item.offload.tokens)} tokens): the model was given a
              preview and saved the full result as{" "}
              <button className="link" onClick={() => onOpenArtifact(item.offload!.artifact_id)}>
                {item.offload.artifact_id}
              </button>
              .
            </p>
          )}
          {item.content != null && (
            <>
              <div className="label">Result{item.offload ? " (full, as the tool returned it)" : ""}</div>
              <pre className={item.status === "error" ? "pre-error" : ""}>{item.content}</pre>
            </>
          )}
        </div>
      )}
    </li>
  );
}
