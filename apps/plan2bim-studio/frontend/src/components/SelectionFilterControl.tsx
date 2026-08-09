import { ListFilter, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { selectionFilterOptions } from "../selectionFilters";
import type { CollectionName } from "../types";

interface SelectionFilterControlProps {
  exclusions: CollectionName[];
  counts: Partial<Record<CollectionName, number>>;
  onToggle: (collection: CollectionName) => void;
  onReset: () => void;
}

export function SelectionFilterControl({
  exclusions,
  counts,
  onToggle,
  onReset,
}: SelectionFilterControlProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div className="selection-filter-control" ref={rootRef}>
      <button
        className={exclusions.length ? "selection-filter-trigger active" : "selection-filter-trigger"}
        aria-label="Selection filters"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
        title="Choose which BIM categories can be selected"
      >
        <ListFilter size={14} />
        <span>FILTER</span>
        <b>{exclusions.length ? `${exclusions.length} OFF` : "ALL"}</b>
      </button>
      {open ? (
        <section className="selection-filter-popover" aria-label="Selection filter menu">
          <header>
            <div>
              <span>SELECTION FILTER</span>
              <strong>{selectionFilterOptions.length - exclusions.length} of {selectionFilterOptions.length} categories active</strong>
            </div>
            <button onClick={onReset} disabled={!exclusions.length} title="Reset selection filters">
              <RotateCcw size={13} /> Reset
            </button>
          </header>
          <div className="selection-filter-options">
            {selectionFilterOptions.map((option) => {
              const enabled = !exclusions.includes(option.collection);
              return (
                <label key={option.collection}>
                  <input
                    type="checkbox"
                    checked={enabled}
                    onChange={() => onToggle(option.collection)}
                  />
                  <span>
                    <strong>{option.label}</strong>
                    <small>{option.detail}</small>
                  </span>
                  <b>{counts[option.collection] ?? 0}</b>
                </label>
              );
            })}
          </div>
          <footer>Applies to 2D Window, Crossing, hover cycling, and 3D picking.</footer>
        </section>
      ) : null}
    </div>
  );
}
