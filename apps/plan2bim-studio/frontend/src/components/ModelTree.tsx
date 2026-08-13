import {
  AlertTriangle,
  Box,
  ChevronDown,
  ChevronsDown,
  ChevronsUp,
  CircleDot,
  DoorOpen,
  Eye,
  EyeOff,
  Layers3,
  Link2,
  Lock,
  LockOpen,
  Scan,
  Search,
  Ruler,
  SquareDashed,
  Warehouse,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { includesSelection, selectionKey } from "../editorViewState";
import { confidenceLabel, entities } from "../graph";
import {
  collapsedModelTree,
  compareModelTreeItems,
  expandedModelTree,
  modelTreeSelectionRange,
  treeSectionIsExpanded,
} from "../modelTreeNavigation";
import type { CollectionName, PlanGraph, Selection } from "../types";
import { reviewPriorityMap, type ReviewPriority } from "../reviewPlanner";
import type { BimSelectionSet } from "../selectionSets";
import { SelectionSetsPanel } from "./SelectionSetsPanel";

const sections: Array<{
  collection: CollectionName;
  label: string;
  icon: typeof Box;
}> = [
  { collection: "walls", label: "Walls", icon: Warehouse },
  { collection: "openings", label: "Doors & windows", icon: DoorOpen },
  { collection: "rooms", label: "Rooms", icon: SquareDashed },
  { collection: "fixtures", label: "Objects", icon: Box },
  { collection: "routes", label: "Building systems", icon: CircleDot },
  { collection: "vertical_connections", label: "Vertical circulation", icon: Layers3 },
  { collection: "constraints", label: "Constraints", icon: Link2 },
  { collection: "dimensions", label: "Dimensions", icon: Ruler },
];

interface ModelTreeProps {
  graph: PlanGraph;
  levelId: string;
  selections: Selection[];
  reviewOnly: boolean;
  reviewPriorities: ReviewPriority[];
  onReviewOnly: (value: boolean) => void;
  onSelect: (selection: Selection, additive?: boolean) => void;
  onSelectMany: (selections: Selection[], additive?: boolean) => void;
  onOpenContextMenu: (selection: Selection, clientX: number, clientY: number) => void;
  hiddenCollections: CollectionName[];
  lockedCollections: CollectionName[];
  hiddenEntities: Selection[];
  lockedEntities: Selection[];
  isolatedEntities: Selection[];
  selectionSets: BimSelectionSet[];
  onToggleVisibility: (collection: CollectionName) => void;
  onToggleLock: (collection: CollectionName) => void;
  onToggleEntityVisibility: (selection: Selection) => void;
  onToggleEntityLock: (selection: Selection) => void;
  onExitIsolation: () => void;
  onShowAll: () => void;
  onUnlockAll: () => void;
  onCreateSelectionSet: (name: string) => void;
  onRecallSelectionSet: (set: BimSelectionSet) => void;
  onIsolateSelectionSet: (set: BimSelectionSet) => void;
  onRenameSelectionSet: (id: string, name: string) => void;
  onDeleteSelectionSet: (id: string) => void;
}

export function ModelTree({
  graph,
  levelId,
  selections,
  reviewOnly,
  reviewPriorities,
  onReviewOnly,
  onSelect,
  onSelectMany,
  onOpenContextMenu,
  hiddenCollections,
  lockedCollections,
  hiddenEntities,
  lockedEntities,
  isolatedEntities,
  selectionSets,
  onToggleVisibility,
  onToggleLock,
  onToggleEntityVisibility,
  onToggleEntityLock,
  onExitIsolation,
  onShowAll,
  onUnlockAll,
  onCreateSelectionSet,
  onRecallSelectionSet,
  onIsolateSelectionSet,
  onRenameSelectionSet,
  onDeleteSelectionSet,
}: ModelTreeProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>(collapsedModelTree);
  const treeScrollRef = useRef<HTMLDivElement | null>(null);
  const rangeAnchorRef = useRef<Selection | null>(null);
  const normalizedQuery = query.trim().toLowerCase();
  const prioritiesByKey = useMemo(
    () => reviewPriorityMap(reviewPriorities),
    [reviewPriorities],
  );
  const visibleSections = useMemo(
    () =>
      sections.map((section) => ({
        ...section,
        items: entities(graph, section.collection).filter((item) => {
          if (section.collection === "vertical_connections") {
            if (item.from_level_id !== levelId && item.to_level_id !== levelId) return false;
          } else if (item.level_id && item.level_id !== levelId) return false;
          if (reviewOnly && !prioritiesByKey.has(selectionKey({ collection: section.collection, id: item.id }))) return false;
          if (!normalizedQuery) return true;
          return `${item.id} ${item.name ?? ""} ${item.type ?? ""} ${item.family_id ?? ""}`
            .toLowerCase()
            .includes(normalizedQuery);
        }).sort((left, right) => {
          if (reviewOnly) {
            const leftRisk = prioritiesByKey.get(selectionKey({ collection: section.collection, id: left.id }))?.score ?? 0;
            const rightRisk = prioritiesByKey.get(selectionKey({ collection: section.collection, id: right.id }))?.score ?? 0;
            if (leftRisk !== rightRisk) return rightRisk - leftRisk;
          }
          return compareModelTreeItems(left, right);
        }),
      })),
    [graph, levelId, normalizedQuery, prioritiesByKey, reviewOnly],
  );
  const visibleCount = visibleSections.reduce((total, section) => total + section.items.length, 0);
  const unresolvedDetectionCount = (graph.detection_review_candidates ?? []).filter(
    (item) => item.level_id === levelId,
  ).length;
  const visibleCounts = Object.fromEntries(
    visibleSections.map((section) => [section.collection, section.items.length]),
  ) as Partial<Record<CollectionName, number>>;

  useEffect(() => {
    const primary = selections.at(-1);
    if (!primary) return;
    rangeAnchorRef.current = primary;
    setOpen((current) => current[primary.collection]
      ? current
      : { ...current, [primary.collection]: true });
  }, [selections]);

  useEffect(() => {
    const primary = selections.at(-1);
    if (!primary || !open[primary.collection]) return;
    const key = `${primary.collection}:${primary.id}`;
    const frame = requestAnimationFrame(() => {
      const row = [...(treeScrollRef.current?.querySelectorAll<HTMLElement>("[data-selection-key]") ?? [])]
        .find((element) => element.dataset.selectionKey === key);
      row?.scrollIntoView({ block: "nearest" });
    });
    return () => cancelAnimationFrame(frame);
  }, [levelId, open, selections]);

  return (
    <aside className="model-tree panel">
      <div className="panel-title-row">
        <div>
          <span className="eyebrow">MODEL BROWSER</span>
          <h2>Project elements</h2>
        </div>
        <Layers3 size={18} />
      </div>
      <label className="search-field">
        <Search size={15} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find ID, type, family" />
      </label>
      <label className="review-filter">
        <input type="checkbox" checked={reviewOnly} onChange={(event) => onReviewOnly(event.target.checked)} />
        <span>Only items requiring review</span>
      </label>
      {unresolvedDetectionCount ? (
        <div className="detection-review-summary" role="status">
          <AlertTriangle size={14} />
          <span><b>{unresolvedDetectionCount}</b> source marks need classification</span>
        </div>
      ) : null}
      <div className="tree-navigation-tools" aria-label="Model Browser navigation">
        <span><b>{visibleCount}</b> elements · Shift range · Ctrl add</span>
        <button
          aria-label="Expand all populated categories"
          title="Expand all populated categories"
          onClick={() => setOpen(expandedModelTree(visibleCounts))}
        >
          <ChevronsDown size={13} />
        </button>
        <button
          aria-label="Collapse all categories"
          title="Collapse all categories"
          onClick={() => setOpen(collapsedModelTree())}
        >
          <ChevronsUp size={13} />
        </button>
      </div>
      {isolatedEntities.length || hiddenCollections.length || hiddenEntities.length || lockedCollections.length || lockedEntities.length ? (
        <div className="browser-view-actions" aria-label="Model Browser recovery actions">
          {isolatedEntities.length ? (
            <button onClick={onExitIsolation} title="Exit 2D and 3D isolation">
              <Scan size={12} /> Exit isolate
            </button>
          ) : null}
          {hiddenCollections.length || hiddenEntities.length ? (
            <button onClick={onShowAll} title="Show every model element">
              <Eye size={12} /> Show all
            </button>
          ) : null}
          {lockedCollections.length || lockedEntities.length ? (
            <button onClick={onUnlockAll} title="Unlock every model element">
              <LockOpen size={12} /> Unlock all
            </button>
          ) : null}
        </div>
      ) : null}
      <SelectionSetsPanel
        sets={selectionSets}
        currentSelectionCount={selections.length}
        onCreate={onCreateSelectionSet}
        onRecall={onRecallSelectionSet}
        onIsolate={onIsolateSelectionSet}
        onRename={onRenameSelectionSet}
        onDelete={onDeleteSelectionSet}
      />
      <div className="tree-scroll" ref={treeScrollRef}>
        {visibleSections.map(({ collection, label, icon: Icon, items }) => {
          const expanded = treeSectionIsExpanded(
            open[collection],
            Boolean(normalizedQuery) || reviewOnly,
            items.length,
          );
          return (
          <section className={`tree-section${hiddenCollections.includes(collection) ? " hidden-category" : ""}${lockedCollections.includes(collection) ? " locked-category" : ""}`} key={collection}>
            <div className="tree-section-header">
              <button
                className="tree-section-expand"
                aria-label={`${expanded ? "Collapse" : "Expand"} ${label}`}
                onClick={() => setOpen((current) => ({ ...current, [collection]: !current[collection] }))}
              >
                <ChevronDown className={expanded ? "" : "closed"} size={15} />
                <Icon size={15} />
                <span>{label}</span>
                <b>{items.length}</b>
              </button>
              <button
                className="tree-state-button"
                aria-label={`${hiddenCollections.includes(collection) ? "Show" : "Hide"} ${label}`}
                title={`${hiddenCollections.includes(collection) ? "Show" : "Hide"} ${label}`}
                onClick={() => onToggleVisibility(collection)}
              >
                {hiddenCollections.includes(collection) ? <EyeOff size={13} /> : <Eye size={13} />}
              </button>
              <button
                className="tree-state-button"
                aria-label={`${lockedCollections.includes(collection) ? "Unlock" : "Lock"} ${label}`}
                title={`${lockedCollections.includes(collection) ? "Unlock" : "Lock"} ${label}`}
                onClick={() => onToggleLock(collection)}
              >
                {lockedCollections.includes(collection) ? <Lock size={13} /> : <LockOpen size={13} />}
              </button>
            </div>
            {expanded ? (
              <div className="tree-items">
                {items.map((item) => {
                  const itemSelection = { collection, id: item.id } as Selection;
                  const active = selections.some(
                    (selection) =>
                      selection.collection === collection && selection.id === item.id,
                  );
                  const itemHidden = includesSelection(hiddenEntities, itemSelection);
                  const itemLocked = includesSelection(lockedEntities, itemSelection);
                  const inheritedLock = lockedCollections.includes(collection);
                  const outsideIsolation = isolatedEntities.length > 0 && !includesSelection(isolatedEntities, itemSelection);
                  const reviewPriority = prioritiesByKey.get(selectionKey(itemSelection));
                  const title = String(item.name ?? item.type ?? item.family_id ?? item.id.split(":").at(-1));
                  return (
                    <div
                      className={`tree-item-row${active ? " active" : ""}${itemHidden ? " item-hidden" : ""}${itemLocked || inheritedLock ? " item-locked" : ""}${outsideIsolation ? " outside-isolation" : ""}`}
                      key={item.id}
                      data-selection-key={`${collection}:${item.id}`}
                    >
                      <button
                        className="tree-item"
                        onClick={(event) => {
                          const additive = event.ctrlKey || event.metaKey;
                          if (event.shiftKey) {
                            const anchorId = rangeAnchorRef.current?.collection === collection
                              ? rangeAnchorRef.current.id
                              : null;
                            onSelectMany(
                              modelTreeSelectionRange(collection, items, anchorId, item.id),
                              additive,
                            );
                          } else {
                            onSelect(itemSelection, additive);
                          }
                          rangeAnchorRef.current = itemSelection;
                        }}
                        onContextMenu={(event) => {
                          event.preventDefault();
                          onOpenContextMenu(itemSelection, event.clientX, event.clientY);
                        }}
                      >
                        <span className={`status-dot ${item.review_state === "accepted" ? "accepted" : "review"}`} />
                        <span className="tree-copy">
                          <strong>{title}</strong>
                          <small>{item.id}</small>
                        </span>
                        <span
                          className={`confidence${reviewOnly && reviewPriority ? ` review-risk ${reviewPriority.band}` : ""}`}
                          title={reviewPriority ? `Review risk ${reviewPriority.percent}/100 · ${reviewPriority.reasons[0]?.label ?? "guided review"}` : undefined}
                        >
                          {reviewOnly && reviewPriority ? `R ${reviewPriority.percent}` : confidenceLabel(item)}
                        </span>
                      </button>
                      <button
                        className="tree-state-button item-state-button"
                        aria-label={`${itemHidden ? "Show" : "Hide"} ${title} ${item.id}`}
                        title={`${itemHidden ? "Show" : "Hide"} ${title}`}
                        onClick={() => onToggleEntityVisibility(itemSelection)}
                      >
                        {itemHidden ? <EyeOff size={12} /> : <Eye size={12} />}
                      </button>
                      <button
                        className="tree-state-button item-state-button"
                        aria-label={inheritedLock ? `${title} locked by ${label}` : `${itemLocked ? "Unlock" : "Lock"} ${title} ${item.id}`}
                        title={inheritedLock ? `Locked by ${label}` : `${itemLocked ? "Unlock" : "Lock"} ${title}`}
                        disabled={inheritedLock}
                        onClick={() => onToggleEntityLock(itemSelection)}
                      >
                        {itemLocked || inheritedLock ? <Lock size={12} /> : <LockOpen size={12} />}
                      </button>
                    </div>
                  );
                })}
                {!items.length ? <p className="empty-tree">No matching elements</p> : null}
              </div>
            ) : null}
          </section>
          );
        })}
      </div>
    </aside>
  );
}
