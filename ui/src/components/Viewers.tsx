import { useEffect, useState } from "react";
import { api } from "../api";
import type { TranscriptMessage } from "../types";
import { Modal } from "./Modal";

/** Full text of an offloaded result or a compaction archive, with a simple find box. */
export function ArtifactViewer({ id, onClose }: { id: string; onClose: () => void }) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  useEffect(() => {
    api.artifact(id).then(setText, (e: Error) => setError(e.message));
  }, [id]);
  const matches = query && text ? text.toLowerCase().split(query.toLowerCase()).length - 1 : 0;
  return (
    <Modal title={<code>{id}</code>} onClose={onClose}>
      {error && <p className="error-text">{error}</p>}
      {text == null && !error && <p className="muted">Loading…</p>}
      {text != null && (
        <>
          <div className="viewer-bar">
            <input placeholder="Find in artifact" value={query} onChange={(e) => setQuery(e.target.value)} />
            <span className="muted">
              {query ? `${matches} match${matches === 1 ? "" : "es"} · ` : ""}
              {text.length.toLocaleString()} characters
            </span>
          </div>
          <pre className="viewer-text">{query ? highlight(text, query) : text}</pre>
        </>
      )}
    </Modal>
  );
}

function highlight(text: string, query: string) {
  const parts: (string | JSX.Element)[] = [];
  const lower = text.toLowerCase();
  const q = query.toLowerCase();
  let i = 0;
  let n = 0;
  while (n < 500) {
    const at = lower.indexOf(q, i);
    if (at === -1) break;
    parts.push(text.slice(i, at), <mark key={n++}>{text.slice(at, at + q.length)}</mark>);
    i = at + q.length;
  }
  parts.push(text.slice(i));
  return parts;
}

/** Every message the harness keeps for this session, marking the part a summary stands in for. */
export function TranscriptViewer({
  sessionId,
  coveredUntil,
  onClose,
}: {
  sessionId: string;
  coveredUntil: number | null;
  onClose: () => void;
}) {
  const [messages, setMessages] = useState<TranscriptMessage[] | null>(null);
  useEffect(() => {
    api.transcript(sessionId).then(setMessages);
  }, [sessionId]);
  return (
    <Modal title="Full transcript (state.messages)" onClose={onClose}>
      <p className="muted">
        Everything the harness has stored for this chat, oldest first. Nothing here is ever deleted.
        {coveredUntil
          ? ` Messages 1–${coveredUntil - 1} are faded: the model no longer sees them directly, only the summary (and the archives) of them.`
          : ""}
      </p>
      {messages == null && <p className="muted">Loading…</p>}
      <ol className="transcript">
        {messages?.map((m) => {
          const summarized = coveredUntil != null && m.seq >= 1 && m.seq < coveredUntil;
          const calls: { name: string; arguments: unknown; raw_arguments?: string }[] =
            (m.body.parts ?? []).filter((p: { type: string }) => p.type === "tool_call").map((p: { call: never }) => p.call);
          return (
            <li key={m.seq} className={`tr-msg role-${m.role}${summarized ? " summarized" : ""}`}>
              <div className="tr-head">
                <span className="tr-seq">#{m.seq}</span>
                <span className="tr-role">{m.role === "tool" ? `tool · ${m.body.name ?? ""}` : m.role}</span>
                {m.body.is_error && <span className="badge badge-err">error</span>}
                {m.body.metadata?.source === "harness" && <span className="badge">harness</span>}
              </div>
              {m.text && <TruncatedText text={m.text} />}
              {calls.map((c, i) => (
                <code key={i} className="tr-call">
                  → {c.name}({JSON.stringify(c.arguments ?? c.raw_arguments ?? {})})
                </code>
              ))}
            </li>
          );
        })}
      </ol>
    </Modal>
  );
}

function TruncatedText({ text, limit = 1200 }: { text: string; limit?: number }) {
  const [full, setFull] = useState(false);
  if (text.length <= limit || full) return <pre className="tr-text">{text}</pre>;
  return (
    <>
      <pre className="tr-text">{text.slice(0, limit)}…</pre>
      <button className="link" onClick={() => setFull(true)}>
        Show all {text.length.toLocaleString()} characters
      </button>
    </>
  );
}
