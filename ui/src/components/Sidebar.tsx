import { useState } from "react";
import type { Incident, OwnerType, SessionSummary, User } from "../types";
import { relativeTime } from "../format";

const SEVERITIES = ["sev1", "sev2", "sev3", "sev4"];

export function Sidebar({
  model,
  ownerType,
  onOwnerType,
  users,
  incidents,
  ownerId,
  onSelectOwner,
  onCreateUser,
  onCreateIncident,
  sessions,
  sessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
}: {
  model: string | undefined;
  ownerType: OwnerType;
  onOwnerType: (t: OwnerType) => void;
  users: User[];
  incidents: Incident[];
  ownerId: string | null;
  onSelectOwner: (id: string) => void;
  onCreateUser: (name: string, email: string) => Promise<void>;
  onCreateIncident: (title: string, severity: string) => Promise<void>;
  sessions: SessionSummary[];
  sessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
}) {
  const [adding, setAdding] = useState(false);
  const owners =
    ownerType === "user"
      ? users.map((u) => ({ id: u.id, label: u.name, sub: u.email ?? "" }))
      : incidents.map((i) => ({ id: i.id, label: i.title, sub: `${i.severity} · ${i.status}` }));

  return (
    <nav className="sidebar">
      <div className="brand">
        <span className="brand-name">Agent Harness</span>
        {model && <span className="chip mono" title="Model">{model}</span>}
      </div>

      <div className="segmented" role="tablist" aria-label="Chat owner type">
        {(["user", "incident"] as const).map((t) => (
          <button key={t} role="tab" aria-selected={ownerType === t} onClick={() => { onOwnerType(t); setAdding(false); }}>
            {t === "user" ? "Users" : "Incidents"}
          </button>
        ))}
      </div>

      <section className="side-section">
        <div className="side-head">
          <h2>{ownerType === "user" ? "Users" : "Incidents"}</h2>
          <button className="icon-btn" onClick={() => setAdding(!adding)} aria-label={`Add ${ownerType}`}>
            {adding ? "✕" : "+"}
          </button>
        </div>
        {adding && (
          <OwnerForm
            ownerType={ownerType}
            onSubmit={async (a, b) => {
              if (ownerType === "user") await onCreateUser(a, b);
              else await onCreateIncident(a, b);
              setAdding(false);
            }}
          />
        )}
        {owners.length === 0 && !adding && (
          <p className="muted small">No {ownerType === "user" ? "users" : "incidents"} yet. Add one with +.</p>
        )}
        <ul className="owner-list">
          {owners.map((o) => (
            <li key={o.id}>
              <button className={o.id === ownerId ? "active" : ""} onClick={() => onSelectOwner(o.id)}>
                <span className="truncate">{o.label}</span>
                {o.sub && <span className="muted small truncate">{o.sub}</span>}
              </button>
            </li>
          ))}
        </ul>
      </section>

      {ownerId && (
        <section className="side-section grow">
          <div className="side-head">
            <h2>Chats</h2>
            <button className="btn btn-small" onClick={onNewSession}>
              + New chat
            </button>
          </div>
          {sessions.length === 0 && <p className="muted small">No chats yet.</p>}
          <ul className="session-list">
            {sessions.map((s) => (
              <li key={s.id} className={s.id === sessionId ? "active" : ""}>
                <button className="session-btn" onClick={() => onSelectSession(s.id)}>
                  <span className="truncate">{s.title}</span>
                  <span className="muted small">{relativeTime(s.updated_at)}</span>
                </button>
                <button
                  className="icon-btn delete"
                  aria-label="Delete chat"
                  onClick={() => confirm(`Delete "${s.title}" and its whole history?`) && onDeleteSession(s.id)}
                >
                  🗑
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </nav>
  );
}

function OwnerForm({ ownerType, onSubmit }: { ownerType: OwnerType; onSubmit: (a: string, b: string) => Promise<void> }) {
  const [a, setA] = useState("");
  const [b, setB] = useState(ownerType === "user" ? "" : "sev3");
  const [error, setError] = useState<string | null>(null);
  return (
    <form
      className="owner-form"
      onSubmit={async (e) => {
        e.preventDefault();
        if (!a.trim()) return;
        try {
          await onSubmit(a.trim(), b.trim());
        } catch (err) {
          setError((err as Error).message);
        }
      }}
    >
      <input autoFocus placeholder={ownerType === "user" ? "Name" : "Incident title"} value={a} onChange={(e) => setA(e.target.value)} />
      {ownerType === "user" ? (
        <input placeholder="Email (optional)" type="email" value={b} onChange={(e) => setB(e.target.value)} />
      ) : (
        <select value={b} onChange={(e) => setB(e.target.value)}>
          {SEVERITIES.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      )}
      {error && <p className="error-text small">{error}</p>}
      <button className="btn btn-small" type="submit" disabled={!a.trim()}>
        Add {ownerType}
      </button>
    </form>
  );
}
