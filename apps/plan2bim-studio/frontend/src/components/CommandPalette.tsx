import { CornerDownLeft, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  nextCommandIndex,
  rankStudioCommands,
  type StudioCommand,
} from "../commandPalette";

interface CommandPaletteProps {
  open: boolean;
  commands: StudioCommand[];
  recentCommandIds: string[];
  onClose: () => void;
  onExecute: (command: StudioCommand) => void;
}

export function CommandPalette({
  open,
  commands,
  recentCommandIds,
  onClose,
  onExecute,
}: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const ranked = useMemo(
    () => rankStudioCommands(commands, query, recentCommandIds),
    [commands, query, recentCommandIds],
  );

  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActiveIndex(0);
  }, [open]);

  useEffect(() => {
    setActiveIndex((current) => Math.max(0, Math.min(current, ranked.length - 1)));
  }, [ranked.length]);

  useEffect(() => {
    if (!open || activeIndex < 0) return;
    listRef.current
      ?.querySelector<HTMLElement>(`[data-command-index="${activeIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, open]);

  if (!open) return null;

  const activeCommand = ranked[activeIndex]?.command;
  const execute = (command: StudioCommand) => {
    if (command.enabled === false) return;
    onExecute(command);
  };

  return (
    <div
      className="command-palette-backdrop"
      role="presentation"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <section className="command-palette" role="dialog" aria-modal="true" aria-label="Command palette">
        <div className="command-palette-search">
          <Search size={17} />
          <input
            autoFocus
            role="combobox"
            aria-expanded="true"
            aria-controls="studio-command-list"
            aria-activedescendant={activeCommand ? `studio-command-${activeCommand.id}` : undefined}
            placeholder="Search commands, tools, and views"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setActiveIndex(0);
            }}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                onClose();
                return;
              }
              const direction = event.key === "ArrowDown"
                ? "next"
                : event.key === "ArrowUp"
                  ? "previous"
                  : event.key === "Home"
                    ? "first"
                    : event.key === "End"
                      ? "last"
                      : null;
              if (direction) {
                event.preventDefault();
                setActiveIndex((current) => nextCommandIndex(current, direction, ranked.length));
                return;
              }
              if (event.key === "Enter" && activeCommand) {
                event.preventDefault();
                execute(activeCommand);
              }
            }}
          />
          <kbd>ESC</kbd>
        </div>
        <div className="command-palette-context">
          <span>{query ? `${ranked.length} matching commands` : "Recent and available commands"}</span>
          <small>Type an action or BIM term</small>
        </div>
        <div ref={listRef} id="studio-command-list" className="command-list" role="listbox">
          {ranked.map(({ command, recent }, index) => {
            const enabled = command.enabled !== false;
            return (
              <button
                key={command.id}
                id={`studio-command-${command.id}`}
                data-command-index={index}
                className={index === activeIndex ? "active" : ""}
                role="option"
                aria-selected={index === activeIndex}
                aria-disabled={!enabled}
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => execute(command)}
              >
                <span className="command-result-copy">
                  <b>{command.label}</b>
                  <small>{enabled ? command.group : command.disabledReason ?? "Unavailable for this selection"}</small>
                </span>
                {recent && !query ? <em>RECENT</em> : null}
                {command.shortcut ? <kbd>{command.shortcut}</kbd> : null}
              </button>
            );
          })}
          {!ranked.length ? (
            <div className="command-empty">
              <Search size={18} />
              <strong>No matching command</strong>
              <span>Try a tool name, BIM element, or outcome.</span>
            </div>
          ) : null}
        </div>
        <footer className="command-palette-footer">
          <span><kbd>↑</kbd><kbd>↓</kbd> Navigate</span>
          <span><CornerDownLeft size={12} /> Run command</span>
          <span><kbd>ESC</kbd> Close</span>
        </footer>
      </section>
    </div>
  );
}
