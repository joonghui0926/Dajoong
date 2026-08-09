import { ChevronDown, ChevronUp, History } from "lucide-react";
import { useEffect, useRef } from "react";

import type { HistoryTimelineEntry } from "../historyTimeline";

interface HistoryTimelineProps {
  entries: HistoryTimelineEntry[];
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onJump: (index: number) => void;
}

export function HistoryTimeline({
  entries,
  collapsed,
  onToggleCollapsed,
  onJump,
}: HistoryTimelineProps) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const current = entries.find((entry) => entry.state === "current") ?? entries.at(-1);

  useEffect(() => {
    if (collapsed || !current) return;
    const frame = requestAnimationFrame(() => {
      const target = [...(trackRef.current?.querySelectorAll<HTMLElement>("[data-history-index]") ?? [])]
        .find((element) => Number(element.dataset.historyIndex) === current.index);
      target?.scrollIntoView({ inline: "center", block: "nearest" });
    });
    return () => cancelAnimationFrame(frame);
  }, [collapsed, current?.index, entries.length]);

  return (
    <section className={`history-timeline${collapsed ? " collapsed" : ""}`} aria-label="BIM edit history">
      <header>
        <History size={14} />
        <div>
          <strong>Edit history</strong>
          <span>{current ? `Step ${current.index + 1} of ${entries.length}` : "No history"}</span>
        </div>
        <button
          aria-label={collapsed ? "Expand edit history" : "Collapse edit history"}
          title={collapsed ? "Expand edit history" : "Collapse edit history"}
          onClick={onToggleCollapsed}
        >
          {collapsed ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
      </header>
      {!collapsed ? (
        <div className="history-track" ref={trackRef} role="list" aria-label="Model states">
          {entries.map((entry) => (
            <div className={`history-step-wrap ${entry.state}`} role="listitem" key={`${entry.index}:${entry.operationCount}`}>
              {entry.index ? <span className="history-connector" /> : null}
              <button
                className={`history-step ${entry.state}`}
                data-history-index={entry.index}
                aria-current={entry.state === "current" ? "step" : undefined}
                aria-label={`${entry.label}. ${entry.detail}. Step ${entry.index + 1} of ${entries.length}`}
                title={`${entry.label}\n${entry.detail}\n${entry.operationCount} audited changes`}
                disabled={entry.state === "current"}
                onClick={() => onJump(entry.index)}
              >
                <span>{entry.index + 1}</span>
                <div>
                  <strong>{entry.label}</strong>
                  <small>{entry.detail}</small>
                </div>
              </button>
            </div>
          ))}
        </div>
      ) : (
        <button className="history-current-summary" onClick={onToggleCollapsed}>
          <strong>{current?.label ?? "Imported model"}</strong>
          <span>{current?.detail ?? "Source conversion"}</span>
        </button>
      )}
    </section>
  );
}
