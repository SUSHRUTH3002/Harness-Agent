import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { AppConfig, Incident, OwnerType, SessionSummary, User } from "./types";
import { Sidebar } from "./components/Sidebar";
import { ChatView } from "./components/ChatView";

// Remembered per browser so a reload reopens the same chat. Storage can be unavailable
// (private windows, blocked site data), so every access is guarded.
const STORAGE_KEY = "agent-harness.selection";
type Selection = { ownerType: OwnerType; ownerId: string | null; sessionId: string | null };

function loadSelection(): Selection {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* ignore */
  }
  return { ownerType: "incident", ownerId: null, sessionId: null };
}

function saveSelection(selection: Selection) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(selection));
  } catch {
    /* ignore */
  }
}

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [selection, setSelection] = useState<Selection>(loadSelection);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { ownerType, ownerId, sessionId } = selection;

  const select = (next: Partial<Selection>) =>
    setSelection((prev) => {
      const merged = { ...prev, ...next };
      saveSelection(merged);
      return merged;
    });

  useEffect(() => {
    Promise.all([api.config(), api.users(), api.incidents()])
      .then(([c, u, i]) => {
        setConfig(c);
        setUsers(u);
        setIncidents(i);
      })
      .catch((e: Error) => setError(`Can't reach the server: ${e.message}. Is it running on :8000?`));
  }, []);

  const refreshSessions = useCallback(async () => {
    if (!ownerId) return setSessions([]);
    try {
      setSessions(await api.sessions(ownerType, ownerId));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [ownerType, ownerId]);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  const owner =
    ownerType === "user" ? users.find((u) => u.id === ownerId) : incidents.find((i) => i.id === ownerId);
  const ownerLabel = owner
    ? ownerType === "user"
      ? `User · ${(owner as User).name}`
      : `Incident · ${(owner as Incident).severity} · ${(owner as Incident).title}`
    : "";
  const sessionVisible = !!sessionId && sessions.some((s) => s.id === sessionId);

  return (
    <div className="app">
      <Sidebar
        model={config?.model}
        ownerType={ownerType}
        onOwnerType={(t) => select({ ownerType: t, ownerId: null, sessionId: null })}
        users={users}
        incidents={incidents}
        ownerId={ownerId}
        onSelectOwner={(id) => select({ ownerId: id, sessionId: null })}
        onCreateUser={async (name, email) => {
          const user = await api.createUser(name, email);
          setUsers((prev) => [...prev, user]);
          select({ ownerId: user.id, sessionId: null });
        }}
        onCreateIncident={async (title, severity) => {
          const incident = await api.createIncident(title, severity);
          setIncidents((prev) => [incident, ...prev]);
          select({ ownerId: incident.id, sessionId: null });
        }}
        sessions={sessions}
        sessionId={sessionId}
        onSelectSession={(id) => select({ sessionId: id })}
        onNewSession={async () => {
          if (!ownerId) return;
          const session = await api.createSession(ownerType, ownerId);
          setSessions((prev) => [session, ...prev]);
          select({ sessionId: session.id });
        }}
        onDeleteSession={async (id) => {
          try {
            await api.deleteSession(id);
            if (id === sessionId) select({ sessionId: null });
            refreshSessions();
          } catch (e) {
            setError((e as Error).message);
          }
        }}
      />
      <main className="main">
        {error && (
          <div className="error-banner" onClick={() => setError(null)}>
            {error}
          </div>
        )}
        {sessionVisible ? (
          <ChatView key={sessionId} sessionId={sessionId!} ownerLabel={ownerLabel} config={config} onSessionChanged={refreshSessions} />
        ) : (
          <div className="placeholder">
            <h2>{ownerId ? "Pick a chat or start a new one" : `Pick or add a ${ownerType === "user" ? "user" : "an incident"}`}</h2>
            <p className="muted">
              Every chat belongs to a user or an incident. Its whole history (messages, tool calls, compaction
              summaries and archives) is saved in Postgres, so you can close the tab and continue later.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
