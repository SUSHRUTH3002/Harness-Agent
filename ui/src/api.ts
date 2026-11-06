import type {
  AppConfig,
  Incident,
  OwnerType,
  SessionDetail,
  SessionSummary,
  TranscriptMessage,
  User,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: init?.body ? { "content-type": "application/json" } : undefined,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, message);
  }
  if (response.status === 204) return undefined as T;
  const type = response.headers.get("content-type") ?? "";
  return (type.includes("application/json") ? response.json() : response.text()) as Promise<T>;
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  config: () => request<AppConfig>("/config"),

  users: () => request<User[]>("/users"),
  createUser: (name: string, email: string) => request<User>("/users", { method: "POST", body: json({ name, email }) }),

  incidents: () => request<Incident[]>("/incidents"),
  createIncident: (title: string, severity: string) =>
    request<Incident>("/incidents", { method: "POST", body: json({ title, severity }) }),

  sessions: (ownerType: OwnerType, ownerId: string) =>
    request<SessionSummary[]>(`/sessions?owner_type=${ownerType}&owner_id=${ownerId}`),
  createSession: (ownerType: OwnerType, ownerId: string) =>
    request<SessionSummary>("/sessions", { method: "POST", body: json({ owner_type: ownerType, owner_id: ownerId }) }),
  session: (id: string) => request<SessionDetail>(`/sessions/${id}`),
  transcript: (id: string) => request<TranscriptMessage[]>(`/sessions/${id}/messages`),
  renameSession: (id: string, title: string) =>
    request<SessionSummary>(`/sessions/${id}`, { method: "PATCH", body: json({ title }) }),
  deleteSession: (id: string) => request<void>(`/sessions/${id}`, { method: "DELETE" }),

  send: (sessionId: string, content: string) =>
    request<{ run_id: string }>(`/sessions/${sessionId}/messages`, { method: "POST", body: json({ content }) }),
  cancel: (runId: string) => request<{ cancelled: boolean }>(`/runs/${runId}/cancel`, { method: "POST" }),
  artifact: (id: string) => request<string>(`/artifacts/${encodeURIComponent(id)}`),
};
