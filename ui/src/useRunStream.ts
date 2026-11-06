import { useEffect, useRef } from "react";
import type { RunEvent } from "./types";

/**
 * Follow a live run over server-sent events (GET /api/runs/{id}/events).
 *
 * The server replays every event so far on connect, so a reconnect would repeat them:
 * events are de-duplicated by `seq`. `onEnd` fires once the run is saved (the server's
 * `end` event), or if the run is no longer live (the stream can't be opened).
 */
export function useRunStream(
  runId: string | null,
  handlers: { onEvent: (event: RunEvent) => void; onEnd: () => void },
) {
  const latest = useRef(handlers);
  latest.current = handlers;

  useEffect(() => {
    if (!runId) return;
    const seen = new Set<number>();
    let ended = false;
    const finish = () => {
      if (ended) return;
      ended = true;
      source.close();
      latest.current.onEnd();
    };

    const source = new EventSource(`/api/runs/${runId}/events`);
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as RunEvent;
      if (seen.has(event.seq)) return;
      seen.add(event.seq);
      latest.current.onEvent(event);
    };
    source.addEventListener("end", finish);
    source.onerror = () => {
      // CLOSED = the server refused the stream (e.g. 404: the run already finished); otherwise
      // the browser is reconnecting by itself.
      if (source.readyState === EventSource.CLOSED) finish();
    };
    return () => {
      ended = true;
      source.close();
    };
  }, [runId]);
}
