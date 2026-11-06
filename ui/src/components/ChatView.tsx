import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api";
import { useRunStream } from "../useRunStream";
import type { AppConfig, Run, RunEvent, SessionDetail } from "../types";
import { seconds, tokens } from "../format";
import { Timeline, buildTimeline } from "./Timeline";
import { MemoryPanel } from "./MemoryPanel";
import { ArtifactViewer, TranscriptViewer } from "./Viewers";

export function ChatView({
  sessionId,
  ownerLabel,
  config,
  onSessionChanged,
}: {
  sessionId: string;
  ownerLabel: string;
  config: AppConfig | null;
  onSessionChanged: () => void;
}) {
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveRunId, setLiveRunId] = useState<string | null>(null);
  const [liveInput, setLiveInput] = useState("");
  const [liveEvents, setLiveEvents] = useState<RunEvent[]>([]);
  const [artifactId, setArtifactId] = useState<string | null>(null);
  const [showTranscript, setShowTranscript] = useState(false);
  const [showMemory, setShowMemory] = useState(true);
  const bottom = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    try {
      const d = await api.session(sessionId);
      setDetail(d);
      setError(null);
      return d;
    } catch (e) {
      setError((e as Error).message);
      return null;
    }
  }, [sessionId]);

  useEffect(() => {
    setDetail(null);
    setLiveRunId(null);
    setLiveEvents([]);
    load().then((d) => {
      // Reopening a chat that is still answering: follow that run live.
      if (d?.active_run_id) {
        setLiveInput(d.runs.find((r) => r.id === d.active_run_id)?.input ?? "");
        setLiveRunId(d.active_run_id);
      }
    });
  }, [load]);

  useRunStream(liveRunId, {
    onEvent: (event) => setLiveEvents((prev) => [...prev, event]),
    onEnd: async () => {
      await load();
      setLiveRunId(null);
      setLiveEvents([]);
      onSessionChanged();
    },
  });

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [liveEvents.length, detail?.runs.length]);

  const send = async (text: string) => {
    setError(null);
    try {
      const { run_id } = await api.send(sessionId, text);
      setLiveInput(text);
      setLiveEvents([]);
      setLiveRunId(run_id);
      onSessionChanged(); // the first message names the chat
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    }
  };

  const runs: Run[] = detail?.runs.map((r) => (r.id === liveRunId ? { ...r, events: liveEvents } : r)) ?? [];
  if (liveRunId && !runs.some((r) => r.id === liveRunId)) {
    runs.push({
      id: liveRunId, session_id: sessionId, input: liveInput, status: "running", output: null, error: null,
      usage: null, steps: null, started_at: new Date().toISOString(), finished_at: null, events: liveEvents,
    });
  }
  const messageCount = detail?.runs.length ?? 0;

  return (
    <div className={`chat-layout${showMemory ? " with-memory" : ""}`}>
      <section className="chat">
        <ChatHeader
          title={detail?.title ?? "…"}
          ownerLabel={ownerLabel}
          onRename={async (title) => {
            await api.renameSession(sessionId, title);
            await load();
            onSessionChanged();
          }}
          showMemory={showMemory}
          onToggleMemory={() => setShowMemory(!showMemory)}
        />
        <div className="messages">
          {!detail && !error && <p className="muted center">Loading…</p>}
          {detail && runs.length === 0 && (
            <div className="empty-chat">
              <h3>Ask the agent something</h3>
              <p className="muted">
                Tools available: {config?.tools.join(", ") || "…"}. Each tool call, result, offload and compaction
                appears below as it happens.
              </p>
            </div>
          )}
          {runs.map((run) => (
            <RunView key={run.id} run={run} live={run.id === liveRunId} onOpenArtifact={setArtifactId} />
          ))}
          <div ref={bottom} />
        </div>
        {error && <div className="error-banner">{error}</div>}
        <Composer
          running={!!liveRunId}
          onSend={send}
          onStop={() => liveRunId && api.cancel(liveRunId)}
        />
      </section>
      {showMemory && detail && (
        <MemoryPanel
          memory={detail.memory}
          config={config}
          messageCount={messageCount}
          onOpenArtifact={setArtifactId}
          onOpenTranscript={() => setShowTranscript(true)}
        />
      )}
      {artifactId && <ArtifactViewer id={artifactId} onClose={() => setArtifactId(null)} />}
      {showTranscript && detail && (
        <TranscriptViewer
          sessionId={sessionId}
          coveredUntil={detail.memory.covered_until}
          onClose={() => setShowTranscript(false)}
        />
      )}
    </div>
  );
}

function ChatHeader({
  title,
  ownerLabel,
  onRename,
  showMemory,
  onToggleMemory,
}: {
  title: string;
  ownerLabel: string;
  onRename: (title: string) => Promise<void>;
  showMemory: boolean;
  onToggleMemory: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(title);
  useEffect(() => setValue(title), [title]);
  const commit = async () => {
    setEditing(false);
    if (value.trim() && value.trim() !== title) await onRename(value.trim());
    else setValue(title);
  };
  return (
    <header className="chat-head">
      <div className="chat-title">
        <span className="muted small">{ownerLabel}</span>
        {editing ? (
          <input
            autoFocus
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
              if (e.key === "Escape") {
                setValue(title);
                setEditing(false);
              }
            }}
          />
        ) : (
          <h1 onClick={() => setEditing(true)} title="Click to rename">
            {title}
          </h1>
        )}
      </div>
      <button className="btn btn-secondary" onClick={onToggleMemory} aria-pressed={showMemory}>
        {showMemory ? "Hide memory" : "Show memory"}
      </button>
    </header>
  );
}

function RunView({ run, live, onOpenArtifact }: { run: Run; live: boolean; onOpenArtifact: (id: string) => void }) {
  const finished = run.events.find((e) => e.type === "run.finished")?.data;
  const status: string = live ? (finished?.status ?? "running") : run.status;
  const output: string | null = finished?.output ?? run.output;
  const error = finished?.error ?? run.error;
  const usage = finished?.usage ?? run.usage;
  const duration: number | undefined = finished?.duration;
  const items = buildTimeline(run.events);
  const toolCount = items.filter((i) => i.kind === "tool").length;
  const compactions = items.filter((i) => i.kind === "compacted").length;
  const steps = finished?.steps ?? run.steps ?? items.filter((i) => i.kind === "step").length;
  const running = status === "running";

  return (
    <article className="run">
      <div className="bubble-user">{run.input}</div>
      <details className="activity" open={running || undefined}>
        <summary>
          <span className={`status-pill status-${status}`}>{running ? "working" : status}</span>
          <span>
            {steps} step{steps === 1 ? "" : "s"} · {toolCount} tool call{toolCount === 1 ? "" : "s"}
            {compactions ? ` · ${compactions} compaction${compactions === 1 ? "" : "s"}` : ""}
            {usage ? ` · ${tokens(usage.input_tokens)} in / ${tokens(usage.output_tokens)} out` : ""}
            {duration != null ? ` · ${seconds(duration)}` : ""}
          </span>
        </summary>
        <Timeline events={run.events} onOpenArtifact={onOpenArtifact} />
      </details>
      {output && (
        <div className="bubble-agent markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{output}</ReactMarkdown>
        </div>
      )}
      {error && !running && (
        <div className={`run-error${status === "cancelled" ? " run-cancelled" : ""}`}>
          <strong>{error.code}</strong> {error.message}
        </div>
      )}
    </article>
  );
}

function Composer({
  running,
  onSend,
  onStop,
}: {
  running: boolean;
  onSend: (text: string) => Promise<boolean>;
  onStop: () => void;
}) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const submit = async () => {
    const value = text.trim();
    if (!value || running || sending) return;
    setSending(true);
    if (await onSend(value)) setText("");
    setSending(false);
  };
  return (
    <div className="composer">
      <textarea
        rows={2}
        placeholder={running ? "The agent is working…" : "Message the agent (Enter to send, Shift+Enter for a new line)"}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      {running ? (
        <button className="btn btn-danger" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button className="btn" onClick={submit} disabled={!text.trim() || sending}>
          Send
        </button>
      )}
    </div>
  );
}
