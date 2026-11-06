-- Schema for the web UI. Applied by the server on every startup, so every statement
-- must be safe to run again (IF NOT EXISTS). Postgres 13+ (gen_random_uuid is built in).

-- Who or what a chat session belongs to. Add more owner tables the same way as these two.
CREATE TABLE IF NOT EXISTS users (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name        text NOT NULL,
    email       text UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS incidents (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title        text NOT NULL,
    severity     text NOT NULL DEFAULT 'sev3',
    status       text NOT NULL DEFAULT 'open',
    description  text,
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- One chat. `owner_id` is a user's id or an incident's id, as `owner_type` says.
-- A single column can't carry a foreign key to two tables, so two generated columns copy
-- owner_id into the matching typed column, and *those* carry the real foreign keys:
-- a session can never point at a user/incident that doesn't exist, and deleting the
-- owner deletes its sessions.
CREATE TABLE IF NOT EXISTS sessions (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_type     text NOT NULL CHECK (owner_type IN ('user', 'incident')),
    owner_id       uuid NOT NULL,
    user_id        uuid GENERATED ALWAYS AS (CASE WHEN owner_type = 'user' THEN owner_id END) STORED
                   REFERENCES users (id) ON DELETE CASCADE,
    incident_id    uuid GENERATED ALWAYS AS (CASE WHEN owner_type = 'incident' THEN owner_id END) STORED
                   REFERENCES incidents (id) ON DELETE CASCADE,
    title          text NOT NULL DEFAULT 'New chat',
    -- The harness AgentState minus messages/tool_results (those live in `messages`):
    -- execution_id, step, usage, extensions (compaction checkpoint, archive index), ...
    state          jsonb,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_owner_idx ON sessions (owner_type, owner_id, updated_at DESC);

-- The full, append-only transcript of a session: every user, assistant and tool message.
-- `body` is the harness Message exactly (lossless, used to rebuild AgentState);
-- role/text are copies for display and querying.
CREATE TABLE IF NOT EXISTS messages (
    session_id  uuid NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    seq         integer NOT NULL,
    id          text NOT NULL,
    run_id      uuid,
    role        text NOT NULL,
    text        text NOT NULL DEFAULT '',
    body        jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, seq)
);

-- One user question and everything the agent did to answer it.
CREATE TABLE IF NOT EXISTS runs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   uuid NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    input        text NOT NULL,
    status       text NOT NULL DEFAULT 'running',
    output       text,
    error        jsonb,
    usage        jsonb,
    steps        integer,
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz
);
CREATE INDEX IF NOT EXISTS runs_session_idx ON runs (session_id, started_at);

-- The live events of a run (tool calls, results, offloads, compactions), kept so the
-- UI can show the same step-by-step view when an old chat is reopened.
CREATE TABLE IF NOT EXISTS run_events (
    run_id   uuid NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    seq      integer NOT NULL,
    type     text NOT NULL,
    step     integer NOT NULL,
    data     jsonb NOT NULL,
    at       timestamptz NOT NULL,
    PRIMARY KEY (run_id, seq)
);
