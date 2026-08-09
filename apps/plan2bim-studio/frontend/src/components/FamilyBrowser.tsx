import { Box, Check, Search, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { fixtureFamilies, type FixtureFamily } from "../families";

interface FamilyBrowserProps {
  open: boolean;
  mode: "insert" | "replace";
  selectionCount: number;
  error?: string;
  onClose: () => void;
  onApply: (family: FixtureFamily) => void;
}

export function FamilyBrowser({
  open,
  mode,
  selectionCount,
  error,
  onClose,
  onApply,
}: FamilyBrowserProps) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All");
  const [activeId, setActiveId] = useState(fixtureFamilies[0].id);
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setCategory("All");
  }, [open]);
  const categories = useMemo(
    () => ["All", ...new Set(fixtureFamilies.map((family) => family.category))],
    [],
  );
  const visible = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return fixtureFamilies.filter((family) => {
      if (category !== "All" && family.category !== category) return false;
      if (!normalized) return true;
      return `${family.name} ${family.type} ${family.discipline} ${family.keywords.join(" ")}`
        .toLowerCase()
        .includes(normalized);
    });
  }, [category, query]);
  const active =
    visible.find((family) => family.id === activeId) ?? visible[0] ?? fixtureFamilies[0];
  if (!open) return null;
  return (
    <div
      className="family-browser-backdrop"
      role="presentation"
      onMouseDown={(event) => event.target === event.currentTarget && onClose()}
    >
      <section className="family-browser" role="dialog" aria-label="BIM family browser">
        <header>
          <div>
            <span className="eyebrow">DAJOONG BIM LIBRARY</span>
            <h2>{mode === "insert" ? "Place a component" : `Replace ${selectionCount} selected`}</h2>
          </div>
          <button className="icon-command" onClick={onClose} aria-label="Close family browser">
            <X size={16} />
          </button>
        </header>
        <div className="family-search-row">
          <label className="search-field family-search">
            <Search size={16} />
            <input
              autoFocus
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search family, type, discipline"
            />
          </label>
          <select value={category} onChange={(event) => setCategory(event.target.value)}>
            {categories.map((item) => (
              <option key={item}>{item}</option>
            ))}
          </select>
        </div>
        <div className="family-browser-body">
          <div className="family-results">
            {visible.map((family) => (
              <button
                key={family.id}
                className={family.id === active.id ? "family-result active" : "family-result"}
                onClick={() => setActiveId(family.id)}
                onDoubleClick={() => onApply(family)}
              >
                <span className={`family-glyph ${family.discipline}`}><Box size={18} /></span>
                <span><strong>{family.name}</strong><small>{family.category}</small></span>
                <code>{family.size_m.map((value) => `${Math.round(value * 1000)}`).join(" × ")} mm</code>
              </button>
            ))}
            {!visible.length ? <p className="family-empty">No matching families</p> : null}
          </div>
          <aside className="family-preview">
            <div className={`family-preview-model ${active.discipline}`}>
              <Box size={52} strokeWidth={1.15} />
              <span>{active.type.replaceAll("-", " ")}</span>
            </div>
            <dl>
              <div><dt>Family ID</dt><dd>{active.id}</dd></div>
              <div><dt>Discipline</dt><dd>{active.discipline}</dd></div>
              <div><dt>Host</dt><dd>{active.mounting}</dd></div>
              <div><dt>Dimensions</dt><dd>{active.size_m.map((value) => `${value.toFixed(2)} m`).join(" × ")}</dd></div>
              <div><dt>Material</dt><dd>{active.material}</dd></div>
            </dl>
          </aside>
        </div>
        <footer>
          <span className={error ? "error" : ""} role={error ? "alert" : undefined}>
            {error ?? "Double click a family to enter guided placement mode"}
          </span>
          <div>
            <button className="secondary-button" onClick={onClose}>Cancel</button>
            <button className="primary-button" onClick={() => onApply(active)} disabled={!visible.length}>
              <Check size={15} /> {mode === "insert" ? "Place family" : "Replace family"}
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}
