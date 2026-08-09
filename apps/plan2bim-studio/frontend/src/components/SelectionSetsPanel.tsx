import { Check, ChevronDown, Pencil, Plus, Scan, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import {
  defaultSelectionSetName,
  selectionSetSummary,
  type BimSelectionSet,
} from "../selectionSets";

interface SelectionSetsPanelProps {
  sets: BimSelectionSet[];
  currentSelectionCount: number;
  onCreate: (name: string) => void;
  onRecall: (set: BimSelectionSet) => void;
  onIsolate: (set: BimSelectionSet) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
}

export function SelectionSetsPanel({
  sets,
  currentSelectionCount,
  onCreate,
  onRecall,
  onIsolate,
  onRename,
  onDelete,
}: SelectionSetsPanelProps) {
  const [open, setOpen] = useState(true);
  const [creating, setCreating] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (creating || editingId) inputRef.current?.focus();
  }, [creating, editingId]);

  const beginCreate = () => {
    if (!currentSelectionCount) return;
    setName(defaultSelectionSetName(sets));
    setEditingId(null);
    setCreating(true);
    setOpen(true);
  };
  const beginRename = (set: BimSelectionSet) => {
    setName(set.name);
    setCreating(false);
    setEditingId(set.id);
  };
  const cancelEdit = () => {
    setCreating(false);
    setEditingId(null);
    setName("");
  };
  const commitEdit = () => {
    const cleaned = name.trim();
    if (!cleaned) return;
    if (creating) onCreate(cleaned);
    else if (editingId) onRename(editingId, cleaned);
    cancelEdit();
  };

  return (
    <section className="selection-sets" aria-label="BIM selection sets">
      <div className="selection-sets-header">
        <button
          className="selection-sets-toggle"
          aria-expanded={open}
          onClick={() => setOpen((current) => !current)}
        >
          <ChevronDown className={open ? "" : "closed"} size={14} />
          <span>Selection sets</span>
          <b>{sets.length}</b>
        </button>
        <button
          className="selection-set-add"
          aria-label="Save current selection as a set"
          title={currentSelectionCount ? `Save ${currentSelectionCount} selected element${currentSelectionCount === 1 ? "" : "s"}` : "Select elements to create a set"}
          disabled={!currentSelectionCount}
          onClick={beginCreate}
        >
          <Plus size={13} />
        </button>
      </div>
      {open ? (
        <div className="selection-set-list">
          {creating ? (
            <form
              className="selection-set-editor"
              onSubmit={(event) => {
                event.preventDefault();
                commitEdit();
              }}
            >
              <input
                ref={inputRef}
                value={name}
                maxLength={64}
                aria-label="Selection set name"
                onChange={(event) => setName(event.target.value)}
                onKeyDown={(event) => event.key === "Escape" && cancelEdit()}
              />
              <button type="submit" aria-label="Save selection set" title="Save selection set"><Check size={11} /></button>
              <button type="button" aria-label="Cancel selection set" title="Cancel" onClick={cancelEdit}><X size={11} /></button>
            </form>
          ) : null}
          {sets.map((set) => (
            <div className="selection-set-row" key={set.id}>
              {editingId === set.id ? (
                <form
                  className="selection-set-editor rename"
                  onSubmit={(event) => {
                    event.preventDefault();
                    commitEdit();
                  }}
                >
                  <input
                    ref={inputRef}
                    value={name}
                    maxLength={64}
                    aria-label={`Rename ${set.name}`}
                    onChange={(event) => setName(event.target.value)}
                    onKeyDown={(event) => event.key === "Escape" && cancelEdit()}
                  />
                  <button type="submit" aria-label={`Save name for ${set.name}`} title="Save name"><Check size={11} /></button>
                  <button type="button" aria-label={`Cancel renaming ${set.name}`} title="Cancel" onClick={cancelEdit}><X size={11} /></button>
                </form>
              ) : (
                <button className="selection-set-main" onClick={() => onRecall(set)} title={`Select ${set.name}`}>
                  <strong>{set.name}</strong>
                  <small>{selectionSetSummary(set)}</small>
                </button>
              )}
              {editingId === set.id ? null : (
                <>
                  <button aria-label={`Isolate ${set.name}`} title={`Select and isolate ${set.name}`} onClick={() => onIsolate(set)}><Scan size={11} /></button>
                  <button aria-label={`Rename ${set.name}`} title={`Rename ${set.name}`} onClick={() => beginRename(set)}><Pencil size={11} /></button>
                  <button aria-label={`Delete ${set.name}`} title={`Delete ${set.name}`} onClick={() => onDelete(set.id)}><Trash2 size={11} /></button>
                </>
              )}
            </div>
          ))}
          {!sets.length && !creating ? (
            <p className="selection-set-empty">Select model elements, then save a reusable set.</p>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
