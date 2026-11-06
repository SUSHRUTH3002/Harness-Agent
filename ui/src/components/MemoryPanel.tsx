import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { AppConfig, Memory } from "../types";
import { relativeTime, tokens } from "../format";

/** What the agent carries between turns of this chat: state, budget, compaction, archives. */
export function MemoryPanel({
  memory,
  config,
  messageCount,
  onOpenArtifact,
  onOpenTranscript,
}: {
  memory: Memory;
  config: AppConfig | null;
  messageCount: number;
  onOpenArtifact: (id: string) => void;
  onOpenTranscript: () => void;
}) {
  const budget = config?.max_context_tokens;
  return (
    <aside className="memory">
      <h2>Memory</h2>
      <dl className="stats">
        <div>
          <dt>Steps</dt>
          <dd>{memory.steps}</dd>
        </div>
        <div>
          <dt>Tokens in / out</dt>
          <dd>
            {tokens(memory.usage?.input_tokens)} / {tokens(memory.usage?.output_tokens)}
          </dd>
        </div>
        <div>
          <dt>Budget</dt>
          <dd>{budget ? `${tokens(budget)} · compacts at ${Math.round((config?.compaction_trigger_ratio ?? 0.8) * 100)}%` : "unlimited"}</dd>
        </div>
        <div>
          <dt>Compactions</dt>
          <dd>{memory.compactions}</dd>
        </div>
      </dl>
      <button className="btn btn-secondary wide" onClick={onOpenTranscript} disabled={!messageCount}>
        View full transcript
      </button>

      <section>
        <h3>Archives</h3>
        {memory.archives.length === 0 ? (
          <p className="muted small">
            None yet. When this chat passes the compaction trigger, the older messages are saved here word for word
            before a summary replaces them.
          </p>
        ) : (
          <ul className="archives">
            {memory.archives.map((a) => (
              <li key={a.artifact_id}>
                <button className="link" onClick={() => onOpenArtifact(a.artifact_id)}>
                  {a.artifact_id}
                </button>
                <span className="muted small">
                  messages {a.messages[0]}–{a.messages[1]} · ~{tokens(a.tokens)} tokens · {a.tools.join(", ") || "no tools"} ·{" "}
                  {relativeTime(a.created_at)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>

      {memory.summary && (
        <section>
          <h3>Current summary</h3>
          <p className="muted small">
            What the model sees in place of messages 1–{(memory.covered_until ?? 1) - 1}.
          </p>
          <div className="summary-box markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{memory.summary}</ReactMarkdown>
          </div>
        </section>
      )}

      {memory.execution_id && (
        <p className="muted small mono">execution {memory.execution_id.slice(0, 8)}</p>
      )}
    </aside>
  );
}
