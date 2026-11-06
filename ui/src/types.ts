// Shapes returned by the FastAPI server (server/app.py).

export type OwnerType = "user" | "incident";

export interface User {
  id: string;
  name: string;
  email: string | null;
  created_at: string;
}

export interface Incident {
  id: string;
  title: string;
  severity: string;
  status: string;
  description: string | null;
  created_at: string;
}

export interface SessionSummary {
  id: string;
  owner_type: OwnerType;
  owner_id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  cost_usd: number | null;
}

// One harness event (agent_harness/events.py), as streamed and as stored in run_events.
export interface RunEvent {
  seq: number;
  type: string;
  step: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: Record<string, any>;
  at: string;
}

export interface RunError {
  code: string;
  message: string;
}

export interface Run {
  id: string;
  session_id: string;
  input: string;
  status: string;
  output: string | null;
  error: RunError | null;
  usage: Usage | null;
  steps: number | null;
  started_at: string;
  finished_at: string | null;
  events: RunEvent[];
}

export interface Archive {
  artifact_id: string;
  messages: [number, number];
  tools: string[];
  tokens: number;
  step: number;
  created_at: string;
}

export interface CompactionRecord {
  at: string;
  step: number;
  reason: string;
  messages: [number, number];
  region_tokens: number;
  summary_tokens: number;
  archive: string | null;
}

export interface Memory {
  execution_id: string | null;
  steps: number;
  usage: Usage | null;
  compactions: number;
  covered_until: number | null;
  summary: string | null;
  archives: Archive[];
  compaction_history: CompactionRecord[];
}

export interface SessionDetail extends SessionSummary {
  runs: Run[];
  active_run_id: string | null;
  memory: Memory;
}

export interface TranscriptMessage {
  seq: number;
  id: string;
  run_id: string | null;
  role: "user" | "assistant" | "tool" | "system";
  text: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  body: any;
  created_at: string;
}

export interface AppConfig {
  model: string;
  provider: string;
  tools: string[];
  max_steps: number;
  max_context_tokens: number | null;
  large_result_handling: boolean;
  compaction: boolean;
  compaction_trigger_ratio: number;
}
