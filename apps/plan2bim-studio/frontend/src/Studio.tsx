import {
  Box,
  BoxSelect,
  BrickWall,
  Check,
  ChevronDown,
  ClipboardCopy,
  ClipboardPaste,
  Columns2,
  DoorOpen,
  Download,
  FileInput,
  LogOut,
  Magnet,
  PanelTop,
  Plus,
  Redo2,
  Ruler,
  Rows2,
  Save,
  Search,
  UserRound,
  Undo2,
  UploadCloud,
} from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { ModelTree } from "./components/ModelTree";
import { ModelViewport, type ModelTransformCommit } from "./components/ModelViewport";
import { PlanViewport } from "./components/PlanViewport";
import { PropertyPanel } from "./components/PropertyPanel";
import { DajoongLogo } from "./components/DajoongLogo";
import { ElementContextMenu } from "./components/ElementContextMenu";
import { HistoryTimeline } from "./components/HistoryTimeline";
import type { PatternMode } from "./components/PatternDialog";
import { SelectionActionBar } from "./components/SelectionActionBar";
import { SelectionFilterControl } from "./components/SelectionFilterControl";
import { authConfigured, authFetch, signOut } from "./auth";
import { studioApiUrl } from "./serverApi";
import { AccountDialog } from "./components/AccountDialog";
import { relatedSelectionGroups, type RelatedSelectionGroup } from "./bimRelations";
import { canArrangeSelection, planArrangement } from "./arrangementPlanner";
import {
  createBimClipboardBundle,
  planBimPaste,
  type BimClipboardBundle,
} from "./bimClipboard";
import { copySelectionsToLevel } from "./crossLevelCopy";
import { canMoveExactly, canRotateExactly, exactRotationChanges, exactTranslationChanges } from "./exactTransform";
import {
  canPattern as canPatternSelection,
  defaultMirrorCoordinates,
  linearArrayPattern,
  mirrorPattern,
  patternCenter,
  type MirrorAxis,
} from "./repetitionCommands";
import type { FixtureFamily } from "./families";
import {
  findNearestValidFixtureCopy,
  validateFixtureEntityChanges,
  validateFixtureTransformChanges,
  validateNewFixtures,
  type FixturePlacement,
} from "./fixturePlacement";
import {
  cornerWallChanges,
  joinWallEndpointChanges,
  type AlignmentMode,
  type DistributionAxis,
  type EntityChanges,
} from "./editorCommands";
import {
  includesSelection,
  sanitizeEntityViewState,
  selectionKey,
  toggleSelectionState,
} from "./editorViewState";
import {
  sanitizeSelectionExclusions,
  toggleSelectionExclusion,
} from "./selectionFilters";
import {
  findAvailableOpeningPlacement,
  toggleDoorHanding,
  toggleDoorSwingSide,
  validateOpeningPlacement,
} from "./openingGeometry";
import { planWallTransform, type WallTransformEntry } from "./wallTransform";
import { planRoomBoundaryTransform } from "./roomBoundary";
import { planOpeningRehost } from "./openingRehost";
import {
  collections,
  deleteEntity,
  downloadJson,
  findEntity,
  graphBounds,
  operationFor,
  updateEntity,
} from "./graph";
import { buildHistoryTimeline, historySnapshots } from "./historyTimeline";
import { planReviewQueue, reviewPriorityMap } from "./reviewPlanner";
import {
  recordRecentCommand,
  sanitizeRecentCommandIds,
  type StudioCommand,
} from "./commandPalette";
import {
  createSelectionSet,
  defaultSelectionSetName,
  renameSelectionSet,
  sanitizeSelectionSets,
  type BimSelectionSet,
} from "./selectionSets";
import type {
  CollectionName,
  CorrectionOperation,
  BaseEntity,
  EditorTool,
  ElementContextMenuRequest,
  OpeningEntity,
  PlanGraph,
  Selection,
  ViewMode,
  WallEntity,
  FixtureEntity,
  GeometricConstraintEntity,
  Measurement,
  LevelEntity,
} from "./types";

const RECENT_COMMANDS_KEY = "dajoong-studio-recent-commands-v1";

const CommandPalette = lazy(async () => ({ default: (await import("./components/CommandPalette")).CommandPalette }));
const ConversionDialog = lazy(async () => ({ default: (await import("./components/ConversionDialog")).ConversionDialog }));
const ExactMoveDialog = lazy(async () => ({ default: (await import("./components/ExactMoveDialog")).ExactMoveDialog }));
const ExactRotateDialog = lazy(async () => ({ default: (await import("./components/ExactRotateDialog")).ExactRotateDialog }));
const FamilyBrowser = lazy(async () => ({ default: (await import("./components/FamilyBrowser")).FamilyBrowser }));
const PatternDialog = lazy(async () => ({ default: (await import("./components/PatternDialog")).PatternDialog }));
const QualityReview = lazy(async () => ({ default: (await import("./components/QualityReview")).QualityReview }));

function loadRecentCommandIds(): string[] {
  try {
    return sanitizeRecentCommandIds(JSON.parse(localStorage.getItem(RECENT_COMMANDS_KEY) ?? "[]"));
  } catch {
    return [];
  }
}

function StudioToolFallback() {
  return <div className="studio-tool-loading" role="status"><span />Opening workspace tool</div>;
}

export interface Snapshot {
  graph: PlanGraph;
  operations: CorrectionOperation[];
}

export interface SessionState {
  source: PlanGraph | null;
  present: Snapshot | null;
  past: Snapshot[];
  future: Snapshot[];
  gesture: {
    before: Snapshot;
    selection: Selection;
    changes: Record<string, unknown>;
    entries?: WallTransformEntry[];
    reason?: string;
  } | null;
}

export type SessionAction =
  | { type: "load"; graph: PlanGraph }
  | { type: "recover"; source: PlanGraph; graph: PlanGraph; operations: CorrectionOperation[] }
  | { type: "resetToSource" }
  | { type: "edit"; selection: Selection; changes: Record<string, unknown> }
  | { type: "batchEdit"; selections: Selection[]; changes: Record<string, unknown> }
  | { type: "batchTransform"; selections: Selection[]; changesById: EntityChanges; reason: string }
  | { type: "constrainWalls"; selections: Selection[]; changesById: EntityChanges; constraint: GeometricConstraintEntity; reason: string }
  | { type: "accept"; selection: Selection }
  | { type: "batchAccept"; selections: Selection[] }
  | { type: "delete"; selection: Selection }
  | { type: "batchDelete"; selections: Selection[] }
  | { type: "add"; collection: CollectionName; entity: Record<string, unknown> }
  | { type: "addMany"; items: Array<{ collection: CollectionName; entity: BaseEntity }>; reason?: string }
  | { type: "beginGesture"; selection: Selection }
  | { type: "previewGesture"; selection: Selection; changes: Record<string, unknown> }
  | { type: "previewTransform"; selection: Selection; entries: WallTransformEntry[]; reason: string }
  | { type: "commitGesture" }
  | { type: "cancelGesture" }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "jumpToHistory"; index: number };

export const initialSessionState: SessionState = {
  source: null,
  present: null,
  past: [],
  future: [],
  gesture: null,
};

function transition(state: SessionState, snapshot: Snapshot): SessionState {
  return {
    ...state,
    present: snapshot,
    past: state.present ? [...state.past.slice(-79), state.present] : state.past,
    future: [],
    gesture: null,
  };
}

export function studioSessionReducer(state: SessionState, action: SessionAction): SessionState {
  if (action.type === "load") {
    const graph = structuredClone(action.graph);
    return {
      source: structuredClone(action.graph),
      present: { graph, operations: [] },
      past: [],
      future: [],
      gesture: null,
    };
  }
  if (action.type === "recover") {
    return {
      source: structuredClone(action.source),
      present: { graph: structuredClone(action.graph), operations: structuredClone(action.operations) },
      past: [],
      future: [],
      gesture: null,
    };
  }
  if (action.type === "resetToSource") {
    if (!state.source) return state;
    return {
      source: structuredClone(state.source),
      present: { graph: structuredClone(state.source), operations: [] },
      past: [],
      future: [],
      gesture: null,
    };
  }
  if (action.type === "undo") {
    const previous = state.past.at(-1);
    if (!previous || !state.present) return state;
    return {
      ...state,
      present: previous,
      past: state.past.slice(0, -1),
      future: [state.present, ...state.future],
      gesture: null,
    };
  }
  if (action.type === "redo") {
    const next = state.future[0];
    if (!next || !state.present) return state;
    return {
      ...state,
      present: next,
      past: [...state.past, state.present],
      future: state.future.slice(1),
      gesture: null,
    };
  }
  if (action.type === "jumpToHistory") {
    if (!state.present || state.gesture) return state;
    const snapshots = historySnapshots(state.past, state.present, state.future);
    if (action.index < 0 || action.index >= snapshots.length || action.index === state.past.length) {
      return state;
    }
    return {
      ...state,
      present: snapshots[action.index],
      past: snapshots.slice(0, action.index),
      future: snapshots.slice(action.index + 1),
      gesture: null,
    };
  }
  if (!state.present) return state;
  if (action.type === "beginGesture") {
    return {
      ...state,
      gesture: {
        before: state.present,
        selection: action.selection,
        changes: {},
      },
    };
  }
  if (action.type === "previewGesture") {
    if (
      !state.gesture ||
      state.gesture.selection.collection !== action.selection.collection ||
      state.gesture.selection.id !== action.selection.id
    ) return state;
    return {
      ...state,
      present: {
        graph: updateEntity(state.present.graph, action.selection, action.changes),
        operations: state.present.operations,
      },
      gesture: {
        ...state.gesture,
        changes: { ...state.gesture.changes, ...action.changes },
      },
    };
  }
  if (action.type === "previewTransform") {
    if (
      !state.gesture ||
      state.gesture.selection.collection !== action.selection.collection ||
      state.gesture.selection.id !== action.selection.id
    ) return state;
    let graph = state.gesture.before.graph;
    for (const entry of action.entries) {
      graph = updateEntity(graph, entry.selection, entry.changes);
    }
    const primaryChanges = action.entries.find(
      (entry) => entry.selection.collection === action.selection.collection
        && entry.selection.id === action.selection.id,
    )?.changes ?? {};
    return {
      ...state,
      present: { graph, operations: state.present.operations },
      gesture: {
        ...state.gesture,
        changes: primaryChanges,
        entries: action.entries,
        reason: action.reason,
      },
    };
  }
  if (action.type === "commitGesture") {
    if (!state.gesture || !Object.keys(state.gesture.changes).length) {
      return { ...state, gesture: null };
    }
    const entries = state.gesture.entries ?? [{
      selection: state.gesture.selection,
      changes: state.gesture.changes,
    }];
    const operations = entries.map((entry) => operationFor(
      entry.selection,
      entry.changes,
      state.gesture?.reason ?? (state.gesture?.entries ? "direct_bim_manipulation" : "direct_manipulation"),
    ));
    return {
      ...state,
      present: {
        graph: state.present.graph,
        operations: [...state.present.operations, ...operations],
      },
      past: [...state.past.slice(-79), state.gesture.before],
      future: [],
      gesture: null,
    };
  }
  if (action.type === "cancelGesture") {
    return state.gesture
      ? { ...state, present: state.gesture.before, gesture: null }
      : state;
  }
  if (action.type === "batchTransform") {
    let graph = state.present.graph;
    const operations = [...state.present.operations];
    for (const selection of action.selections) {
      const changes = action.changesById[selection.id];
      if (!changes) continue;
      graph = updateEntity(graph, selection, changes);
      operations.push(operationFor(selection, changes, action.reason));
    }
    return transition(state, { graph, operations });
  }
  if (action.type === "constrainWalls") {
    let graph = state.present.graph;
    const operations = [...state.present.operations];
    for (const selection of action.selections) {
      const changes = action.changesById[selection.id];
      if (!changes) continue;
      graph = updateEntity(graph, selection, changes);
      operations.push(operationFor(selection, changes, action.reason));
    }
    graph = structuredClone(graph);
    graph.constraints = [...(graph.constraints ?? []), action.constraint];
    operations.push({
      id: `add-${Date.now()}-${action.constraint.id}`,
      action: "add",
      collection: "constraints",
      entity_id: action.constraint.id,
      changes: Object.fromEntries(
        Object.entries(action.constraint).filter(([key]) => key !== "id"),
      ),
      reason: action.reason,
    });
    return transition(state, { graph, operations });
  }
  if (action.type === "batchEdit") {
    let graph = state.present.graph;
    const operations = [...state.present.operations];
    for (const selection of action.selections) {
      graph = updateEntity(graph, selection, action.changes);
      operations.push(operationFor(selection, action.changes, "batch_edit"));
    }
    return transition(state, { graph, operations });
  }
  if (action.type === "batchAccept") {
    let graph = state.present.graph;
    const operations = [...state.present.operations];
    for (const selection of action.selections) {
      graph = updateEntity(graph, selection, {});
      operations.push({
        id: `accept-${Date.now()}-${selection.id}`,
        action: "accept",
        collection: selection.collection,
        entity_id: selection.id,
        changes: {},
        reason: "batch_review",
      });
    }
    return transition(state, { graph, operations });
  }
  if (action.type === "batchDelete") {
    let graph = state.present.graph;
    const operations = [...state.present.operations];
    for (const selection of action.selections) {
      graph = deleteEntity(graph, selection);
      operations.push({
        id: `delete-${Date.now()}-${selection.id}`,
        action: "delete",
        collection: selection.collection,
        entity_id: selection.id,
        changes: {},
        reason: "batch_false_positive",
      });
    }
    return transition(state, { graph, operations });
  }
  if (action.type === "edit") {
    const operation = operationFor(action.selection, action.changes);
    const operations = [...state.present.operations];
    const last = operations.at(-1);
    if (last?.action === "update" && last.collection === operation.collection && last.entity_id === operation.entity_id) {
      operations[operations.length - 1] = { ...operation, id: last.id, changes: { ...last.changes, ...operation.changes } };
    } else {
      operations.push(operation);
    }
    return transition(state, { graph: updateEntity(state.present.graph, action.selection, action.changes), operations });
  }
  if (action.type === "accept") {
    const operation: CorrectionOperation = {
      id: `accept-${Date.now()}`,
      action: "accept",
      collection: action.selection.collection,
      entity_id: action.selection.id,
      changes: {},
      reason: "visual_review",
    };
    return transition(state, { graph: updateEntity(state.present.graph, action.selection, {}), operations: [...state.present.operations, operation] });
  }
  if (action.type === "delete") {
    const operation: CorrectionOperation = {
      id: `delete-${Date.now()}`,
      action: "delete",
      collection: action.selection.collection,
      entity_id: action.selection.id,
      changes: {},
      reason: "false_positive",
    };
    return transition(state, { graph: deleteEntity(state.present.graph, action.selection), operations: [...state.present.operations, operation] });
  }
  if (action.type === "addMany") {
    const nextGraph = structuredClone(state.present.graph);
    const operations = [...state.present.operations];
    for (const item of action.items) {
      const record = nextGraph as unknown as Record<CollectionName, BaseEntity[] | undefined>;
      const target = record[item.collection] ?? (record[item.collection] = []);
      target.push(item.entity);
      operations.push({
        id: `add-${Date.now()}-${item.entity.id}`,
        action: "add",
        collection: item.collection,
        entity_id: item.entity.id,
        changes: correctionChangesForAddedEntity(item.entity),
        reason: action.reason ?? "duplicate_component",
      });
    }
    return transition(state, { graph: nextGraph, operations });
  }
  const nextGraph = structuredClone(state.present.graph);
  const record = nextGraph as unknown as Record<CollectionName, BaseEntity[] | undefined>;
  const target = record[action.collection] ?? (record[action.collection] = []);
  target.push(action.entity as BaseEntity);
  const entityId = String(action.entity.id);
  const changes = correctionChangesForAddedEntity(action.entity);
  const operation: CorrectionOperation = {
    id: `add-${Date.now()}`,
    action: "add",
    collection: action.collection,
    entity_id: entityId,
    changes,
    reason: "missing_element",
  };
  return transition(state, { graph: nextGraph, operations: [...state.present.operations, operation] });
}

export function Studio() {
  const [session, rawDispatch] = useReducer(studioSessionReducer, initialSessionState);
  const [selections, setSelections] = useState<Selection[]>([]);
  const [levelId, setLevelId] = useState("L1");
  const [viewMode, setViewMode] = useState<ViewMode>("split");
  const [reviewOnly, setReviewOnly] = useState(false);
  const [sourceUrl, setSourceUrl] = useState<string>("/sample/source.png");
  const [notice, setNotice] = useState("Loading reviewed sample…");
  const [conversionOpen, setConversionOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [commandOpen, setCommandOpen] = useState(false);
  const [recentCommandIds, setRecentCommandIds] = useState(loadRecentCommandIds);
  const [jobId, setJobId] = useState("");
  const [snapIncrementM, setSnapIncrementM] = useState(0.05);
  const [activeTool, setActiveTool] = useState<EditorTool>("select");
  const [familyBrowserOpen, setFamilyBrowserOpen] = useState(false);
  const [familyBrowserMode, setFamilyBrowserMode] = useState<"insert" | "replace">("insert");
  const [familyBrowserError, setFamilyBrowserError] = useState("");
  const [placementFamily, setPlacementFamily] = useState<FixtureFamily | null>(null);
  const [qualityOpen, setQualityOpen] = useState(false);
  const [exactMoveOpen, setExactMoveOpen] = useState(false);
  const [exactRotateOpen, setExactRotateOpen] = useState(false);
  const [patternOpen, setPatternOpen] = useState<PatternMode | null>(null);
  const [historyCollapsed, setHistoryCollapsed] = useState(
    () => localStorage.getItem("dajoong-history-collapsed-v1") === "1",
  );
  const [discardConfirmOpen, setDiscardConfirmOpen] = useState(false);
  const [hiddenCollections, setHiddenCollections] = useState<CollectionName[]>([]);
  const [lockedCollections, setLockedCollections] = useState<CollectionName[]>([]);
  const [hiddenEntities, setHiddenEntities] = useState<Selection[]>([]);
  const [lockedEntities, setLockedEntities] = useState<Selection[]>([]);
  const [isolatedEntities, setIsolatedEntities] = useState<Selection[]>([]);
  const [selectionExclusions, setSelectionExclusions] = useState<CollectionName[]>([]);
  const [contextMenu, setContextMenu] = useState<ElementContextMenuRequest | null>(null);
  const [rehostOpeningId, setRehostOpeningId] = useState<string | null>(null);
  const [bimClipboard, setBimClipboard] = useState<BimClipboardBundle | null>(null);
  const [selectionSets, setSelectionSets] = useState<BimSelectionSet[]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const rehostViewStateRef = useRef<{
    hiddenCollections: CollectionName[];
    hiddenEntities: Selection[];
    isolatedEntities: Selection[];
  } | null>(null);
  const graph = session.present?.graph ?? null;
  const operations = session.present?.operations ?? [];
  const historyEntries = useMemo(
    () => session.present
      ? buildHistoryTimeline(session.past, session.present, session.future)
      : [],
    [session.future, session.past, session.present],
  );
  const selection = selections.at(-1) ?? null;
  const toggleHistoryCollapsed = useCallback(() => {
    setHistoryCollapsed((current) => {
      const next = !current;
      localStorage.setItem("dajoong-history-collapsed-v1", next ? "1" : "0");
      return next;
    });
  }, []);
  const jumpToHistory = useCallback((index: number) => {
    const entry = historyEntries[index];
    if (!entry || entry.state === "current") return;
    rawDispatch({ type: "jumpToHistory", index });
    setContextMenu(null);
    setActiveTool("select");
    setNotice(`Restored history step ${index + 1} · ${entry.label}`);
  }, [historyEntries]);
  const dispatch = useCallback((action: SessionAction) => {
    const impacted = actionMutationSelections(action, graph);
    const affected = [...new Set([
      ...actionCollections(action),
      ...impacted.map((item) => item.collection),
    ])];
    const blocked = affected.find((collection) => lockedCollections.includes(collection));
    if (blocked) {
      setNotice(`${blocked.replaceAll("_", " ")} are locked · command blocked`);
      return;
    }
    const blockedEntity = impacted.find((item) =>
      includesSelection(lockedEntities, item),
    );
    if (blockedEntity) {
      setNotice(`${blockedEntity.id} is locked · command blocked`);
      return;
    }
    rawDispatch(action);
  }, [graph, lockedCollections, lockedEntities]);
  const setSelection = useCallback(
    (next: Selection | null) => setSelections(next ? [next] : []),
    [],
  );

  useEffect(() => {
    if (!graph) {
      setSelections([]);
      return;
    }
    setSelections((current) => {
      const valid = current.filter((item) => findEntity(graph, item) !== null);
      return valid.length === current.length ? current : valid;
    });
    setHiddenEntities((current) => sanitizeEntityViewState(current, graph));
    setLockedEntities((current) => sanitizeEntityViewState(current, graph));
    setIsolatedEntities((current) => sanitizeEntityViewState(current, graph));
    setSelectionSets((current) => sanitizeSelectionSets(current, graph));
  }, [graph]);

  useEffect(() => {
    if (!graph || graph.levels.some((level) => level.id === levelId)) return;
    setLevelId(graph.levels[0]?.id ?? "L1");
    setSelection(null);
  }, [graph, levelId, setSelection]);

  useEffect(() => {
    if (activeTool !== "object" && placementFamily) setPlacementFamily(null);
  }, [activeTool, placementFamily]);

  useEffect(() => {
    const saved = localStorage.getItem("dajoong-plan2bim-studio-session-v1");
    if (saved) {
      try {
        const payload = JSON.parse(saved) as {
          source?: PlanGraph;
          graph?: PlanGraph;
          operations?: CorrectionOperation[];
          measurements?: Measurement[];
          view_state?: {
            hidden_collections?: CollectionName[];
            locked_collections?: CollectionName[];
            hidden_entities?: Selection[];
            locked_entities?: Selection[];
            isolated_entities?: Selection[];
            selection_exclusions?: CollectionName[];
            selection_sets?: BimSelectionSet[];
          };
        };
        if (payload.graph && Array.isArray(payload.graph.walls) && Array.isArray(payload.graph.rooms)) {
          const recoveredGraph = structuredClone(payload.graph);
          if (!Array.isArray(recoveredGraph.dimensions)) {
            recoveredGraph.dimensions = payload.measurements ?? [];
          }
          dispatch({
            type: "recover",
            source: payload.source ?? payload.graph,
            graph: recoveredGraph,
            operations: payload.operations ?? [],
          });
          setHiddenCollections(
            (payload.view_state?.hidden_collections ?? []).filter((collection) => collections.includes(collection)),
          );
          setLockedCollections(
            (payload.view_state?.locked_collections ?? []).filter((collection) => collections.includes(collection)),
          );
          setHiddenEntities(sanitizeEntityViewState(payload.view_state?.hidden_entities, recoveredGraph));
          setLockedEntities(sanitizeEntityViewState(payload.view_state?.locked_entities, recoveredGraph));
          setIsolatedEntities(sanitizeEntityViewState(payload.view_state?.isolated_entities, recoveredGraph));
          setSelectionExclusions(sanitizeSelectionExclusions(payload.view_state?.selection_exclusions));
          setSelectionSets(sanitizeSelectionSets(payload.view_state?.selection_sets, recoveredGraph));
          setLevelId(recoveredGraph.levels[0]?.id ?? "L1");
          setNotice(`Recovered local session · ${payload.operations?.length ?? 0} audited changes`);
          return;
        }
      } catch {
        localStorage.removeItem("dajoong-plan2bim-studio-session-v1");
      }
    }
    fetch("/sample/03-plan-graph.json")
      .then((response) => {
        if (!response.ok) throw new Error(`sample returned ${response.status}`);
        return response.json();
      })
      .then((payload: PlanGraph) => {
        dispatch({ type: "load", graph: payload });
        setLevelId(payload.levels[0]?.id ?? "L1");
        setNotice("Sample loaded · select any plan or model element");
      })
      .catch((error: Error) => setNotice(`Import a PlanGraph to begin · ${error.message}`));
  }, []);

  useEffect(() => {
    if (!session.present) return;
    localStorage.setItem(
      "dajoong-plan2bim-studio-session-v1",
      JSON.stringify({
        source: session.source,
        graph: session.present.graph,
        operations: session.present.operations,
        view_state: {
          hidden_collections: hiddenCollections,
          locked_collections: lockedCollections,
          hidden_entities: hiddenEntities,
          locked_entities: lockedEntities,
          isolated_entities: isolatedEntities,
          selection_exclusions: selectionExclusions,
          selection_sets: selectionSets,
        },
      }),
    );
  }, [hiddenCollections, hiddenEntities, isolatedEntities, lockedCollections, lockedEntities, selectionExclusions, selectionSets, session.present, session.source]);

  useEffect(() => {
    if (!jobId || !levelId) return;
    const controller = new AbortController();
    authFetch(
      studioApiUrl(`/api/jobs/${jobId}/artifacts/render?level_id=${encodeURIComponent(levelId)}`),
      { signal: controller.signal },
    )
      .then(async (response) => {
        if (!response.ok) throw new Error(`source evidence returned ${response.status}`);
        return URL.createObjectURL(await response.blob());
      })
      .then((nextUrl) => {
        if (controller.signal.aborted) {
          URL.revokeObjectURL(nextUrl);
          return;
        }
        setSourceUrl((current) => {
          if (current.startsWith("blob:")) URL.revokeObjectURL(current);
          return nextUrl;
        });
      })
      .catch((error: Error) => {
        if (error.name !== "AbortError") {
          setNotice(`3D model loaded · source evidence unavailable for ${levelId}`);
        }
      });
    return () => controller.abort();
  }, [jobId, levelId]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTyping =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.tagName === "SELECT" ||
        target?.isContentEditable;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandOpen((value) => !value);
        return;
      }
      if (event.key === "Escape" && contextMenu) {
        event.preventDefault();
        setContextMenu(null);
        return;
      }
      if (isTyping) return;
      if (event.key === "Escape") {
        dispatch({ type: "cancelGesture" });
        setCommandOpen(false);
        setFamilyBrowserOpen(false);
        setQualityOpen(false);
        setExactMoveOpen(false);
        setActiveTool("select");
        setSelection(null);
        return;
      }
      if (!(event.ctrlKey || event.metaKey)) return;
      if (event.key.toLowerCase() === "a" && graph) {
        event.preventDefault();
        if (event.shiftKey) {
          setSelection(null);
          return;
        }
        const hiddenKeys = new Set(hiddenEntities.map(selectionKey));
        const selectable = [
          ...graph.walls.filter((item) => item.level_id === levelId).map((item) => ({ collection: "walls" as const, id: item.id })),
          ...graph.openings.filter((item) => item.level_id === levelId).map((item) => ({ collection: "openings" as const, id: item.id })),
          ...graph.rooms.filter((item) => item.level_id === levelId).map((item) => ({ collection: "rooms" as const, id: item.id })),
          ...graph.fixtures.filter((item) => item.level_id === levelId).map((item) => ({ collection: "fixtures" as const, id: item.id })),
          ...graph.routes.filter((item) => item.level_id === levelId).map((item) => ({ collection: "routes" as const, id: item.id })),
          ...(graph.vertical_connections ?? []).filter((item) => item.from_level_id === levelId || item.to_level_id === levelId).map((item) => ({ collection: "vertical_connections" as const, id: item.id })),
        ];
        setSelections(selectable.filter(
          (item) =>
            !hiddenCollections.includes(item.collection) &&
            !selectionExclusions.includes(item.collection) &&
            !hiddenKeys.has(selectionKey(item)),
        ));
        return;
      }
      if (event.key.toLowerCase() === "z") {
        event.preventDefault();
        dispatch({ type: event.shiftKey ? "redo" : "undo" });
      }
      if (event.key.toLowerCase() === "y") {
        event.preventDefault();
        dispatch({ type: "redo" });
      }
      if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        void exportPatch();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const selectedEntity = graph ? findEntity(graph, selection) : null;
  const selectedEntities: BaseEntity[] = graph
    ? selections
        .map((item) => findEntity(graph, item))
        .filter((item): item is BaseEntity => item !== null)
    : [];
  const reviewPriorities = useMemo(() => (graph ? planReviewQueue(graph) : []), [graph]);
  const reviewPrioritiesByKey = useMemo(
    () => reviewPriorityMap(reviewPriorities),
    [reviewPriorities],
  );
  const queue = useMemo(
    () => reviewPriorities.map((priority) => priority.selection),
    [reviewPriorities],
  );
  const selectedReviewPriority = selection
    ? reviewPrioritiesByKey.get(selectionKey(selection)) ?? null
    : null;
  const selectionFilterCounts = useMemo<Partial<Record<CollectionName, number>>>(() => {
    if (!graph) return {};
    return {
      walls: graph.walls.filter((item) => item.level_id === levelId).length,
      openings: graph.openings.filter((item) => item.level_id === levelId).length,
      rooms: graph.rooms.filter((item) => item.level_id === levelId).length,
      fixtures: graph.fixtures.filter((item) => item.level_id === levelId).length,
      routes: graph.routes.filter((item) => item.level_id === levelId).length,
      vertical_connections: (graph.vertical_connections ?? []).filter(
        (item) => item.from_level_id === levelId || item.to_level_id === levelId,
      ).length,
      dimensions: (graph.dimensions ?? []).filter((item) => item.level_id === levelId).length,
      constraints: (graph.constraints ?? []).filter((item) => item.level_id === levelId).length,
    };
  }, [graph, levelId]);
  const isSelectionLocked = useCallback(
    (item: Selection) =>
      lockedCollections.includes(item.collection) || includesSelection(lockedEntities, item),
    [lockedCollections, lockedEntities],
  );
  const isSelectionHidden = useCallback(
    (item: Selection) =>
      hiddenCollections.includes(item.collection) || includesSelection(hiddenEntities, item),
    [hiddenCollections, hiddenEntities],
  );
  const toggleCollectionVisibility = useCallback((collection: CollectionName) => {
    const willHide = !hiddenCollections.includes(collection);
    setHiddenCollections(
      willHide
        ? [...hiddenCollections, collection]
        : hiddenCollections.filter((item) => item !== collection),
    );
    if (willHide) {
      setSelections((selected) => selected.filter((item) => item.collection !== collection));
    }
    setNotice(`${collection.replaceAll("_", " ")} ${willHide ? "hidden" : "shown"} in 2D and 3D`);
  }, [hiddenCollections]);
  const toggleSelectionFilter = useCallback((collection: CollectionName) => {
    const willExclude = !selectionExclusions.includes(collection);
    setSelectionExclusions(toggleSelectionExclusion(selectionExclusions, collection));
    setNotice(`${collection.replaceAll("_", " ")} ${willExclude ? "excluded from" : "restored to"} 2D and 3D selection`);
  }, [selectionExclusions]);
  const toggleCollectionLock = useCallback((collection: CollectionName) => {
    const willLock = !lockedCollections.includes(collection);
    setLockedCollections(
      willLock
        ? [...lockedCollections, collection]
        : lockedCollections.filter((item) => item !== collection),
    );
    if (willLock) {
      rawDispatch({ type: "cancelGesture" });
      if (editorToolCollection(activeTool) === collection) setActiveTool("select");
    }
    setNotice(`${collection.replaceAll("_", " ")} ${willLock ? "locked" : "unlocked"}`);
  }, [activeTool, lockedCollections]);
  const toggleEntityVisibility = useCallback((item: Selection) => {
    const willHide = !includesSelection(hiddenEntities, item);
    setHiddenEntities(toggleSelectionState(hiddenEntities, item));
    setNotice(`${item.id} ${willHide ? "hidden" : "shown"} in 2D and 3D`);
  }, [hiddenEntities]);
  const toggleEntityLock = useCallback((item: Selection) => {
    const willLock = !includesSelection(lockedEntities, item);
    setLockedEntities(toggleSelectionState(lockedEntities, item));
    if (willLock) rawDispatch({ type: "cancelGesture" });
    setNotice(`${item.id} ${willLock ? "locked" : "unlocked"}`);
  }, [lockedEntities]);
  const isolateItems = useCallback((items: Selection[]) => {
    if (!graph || !items.length) return;
    const next = items.filter((item) => findEntity(graph, item));
    if (!next.length) return;
    const selectedKeys = new Set(next.map(selectionKey));
    const selectedCollections = new Set(next.map((item) => item.collection));
    setIsolatedEntities(next);
    setHiddenCollections((current) => current.filter(
      (collection) => !selectedCollections.has(collection),
    ));
    setHiddenEntities((current) => current.filter(
      (item) => !selectedKeys.has(selectionKey(item)),
    ));
    setNotice(`${next.length} element${next.length === 1 ? "" : "s"} isolated in 2D and 3D`);
  }, [graph]);
  const isolateSelection = useCallback(() => {
    isolateItems(selections);
  }, [isolateItems, selections]);
  const saveCurrentSelectionSet = useCallback((name?: string) => {
    const next = createSelectionSet(
      selectionSets,
      name ?? defaultSelectionSetName(selectionSets),
      selections,
    );
    if (!next) {
      setNotice("Select one or more BIM elements before creating a selection set");
      return;
    }
    setSelectionSets((current) => [...current, next]);
    setNotice(`${next.name} saved · ${next.selections.length} reusable element${next.selections.length === 1 ? "" : "s"}`);
  }, [selectionSets, selections]);
  const renameSavedSelectionSet = useCallback((id: string, name: string) => {
    setSelectionSets((current) => renameSelectionSet(current, id, name));
    setNotice(`${name.trim()} renamed`);
  }, []);
  const deleteSavedSelectionSet = useCallback((id: string) => {
    setSelectionSets((current) => {
      const target = current.find((item) => item.id === id);
      if (target) setNotice(`${target.name} removed from selection sets`);
      return current.filter((item) => item.id !== id);
    });
  }, []);
  const activateSelectionSet = useCallback((set: BimSelectionSet, isolate: boolean) => {
    if (!graph) return;
    const valid = sanitizeEntityViewState(set.selections, graph);
    if (!valid.length) {
      setSelectionSets((current) => current.filter((item) => item.id !== set.id));
      setNotice(`${set.name} no longer contains model elements`);
      return;
    }
    const primary = valid.at(-1)!;
    const entity = findEntity(graph, primary);
    const targetLevelId = primary.collection === "levels"
      ? primary.id
      : String(
          entity?.level_id
          ?? (primary.collection === "vertical_connections" ? entity?.from_level_id : "")
          ?? "",
        );
    if (targetLevelId && graph.levels.some((level) => level.id === targetLevelId)) {
      setLevelId(targetLevelId);
    }
    setActiveTool("select");
    setSelections(valid);
    if (isolate) isolateItems(valid);
    setNotice(`${set.name} · ${valid.length} selected${isolate ? " and isolated" : ""}`);
  }, [graph, isolateItems]);
  const recallSelectionSet = useCallback(
    (set: BimSelectionSet) => activateSelectionSet(set, false),
    [activateSelectionSet],
  );
  const isolateSelectionSet = useCallback(
    (set: BimSelectionSet) => activateSelectionSet(set, true),
    [activateSelectionSet],
  );
  const exitIsolation = useCallback(() => {
    setIsolatedEntities([]);
    setNotice("Isolation cleared · full model context restored");
  }, []);
  const showAllElements = useCallback(() => {
    setHiddenCollections([]);
    setHiddenEntities([]);
    setIsolatedEntities([]);
    setNotice("All model elements shown in 2D and 3D");
  }, []);
  const unlockAllElements = useCallback(() => {
    rawDispatch({ type: "cancelGesture" });
    setLockedCollections([]);
    setLockedEntities([]);
    setNotice("All model elements unlocked");
  }, []);
  const revealCollection = useCallback((collection: CollectionName) => {
    setHiddenCollections((current) => current.filter((item) => item !== collection));
  }, []);
  const locateFindingEntities = useCallback((entityIds: string[]) => {
    if (!graph) return;
    for (const entityId of entityIds) {
      for (const collection of collections) {
        const next: Selection = { collection, id: entityId };
        const entity = findEntity(graph, next);
        if (!entity) continue;
        const targetLevelId = collection === "levels"
          ? entity.id
          : String(
              entity.level_id
              ?? (collection === "vertical_connections" ? entity.from_level_id : "")
              ?? "",
            );
        if (targetLevelId && graph.levels.some((level) => level.id === targetLevelId)) {
          setLevelId(targetLevelId);
        }
        setHiddenCollections((current) => current.filter((item) => item !== collection));
        setHiddenEntities((current) => current.filter(
          (item) => item.collection !== collection || item.id !== entityId,
        ));
        setIsolatedEntities((current) => (
          current.length && !includesSelection(current, next) ? [...current, next] : current
        ));
        setSelections([next]);
        setNotice(`${entityId} located from model integrity report`);
        return;
      }
    }
    setNotice("This integrity finding has no selectable model element");
  }, [graph]);
  const onSelect = useCallback((next: Selection, additive = false) => {
    setContextMenu(null);
    setNotice(
      additive
        ? "Selection updated"
        : `${next.id} selected${isSelectionLocked(next) ? " · locked" : ""}${isSelectionHidden(next) ? " · hidden" : ""}`,
    );
    if (!additive) {
      setSelections([next]);
      return;
    }
    setSelections((current) => {
      const index = current.findIndex(
        (item) => item.collection === next.collection && item.id === next.id,
      );
      if (index >= 0) return current.filter((_, itemIndex) => itemIndex !== index);
      return [...current, next];
    });
  }, [isSelectionHidden, isSelectionLocked]);
  const onSelectMany = useCallback((next: Selection[], additive = false) => {
    setContextMenu(null);
    setSelections((current) => {
      if (!additive) return next;
      const keys = new Set(current.map(selectionKey));
      return [...current, ...next.filter((item) => !keys.has(selectionKey(item)))];
    });
    setNotice(
      next.length
        ? `${next.length} element${next.length === 1 ? "" : "s"} captured by area selection${additive ? " · added" : ""}`
        : additive
          ? "Area selection found no additional elements"
          : "Area selection cleared",
    );
  }, []);
  const onSelectTreeRange = useCallback((next: Selection[], additive = false) => {
    setContextMenu(null);
    setSelections((current) => {
      if (!additive) return next;
      const keys = new Set(current.map(selectionKey));
      return [...current, ...next.filter((item) => !keys.has(selectionKey(item)))];
    });
    setNotice(
      next.length > 1
        ? `${next.length} contiguous elements selected in Model Browser${additive ? " · added to selection" : ""}`
        : next.length === 1
          ? `${next[0].id} selected in Model Browser`
          : "No matching Model Browser elements",
    );
  }, []);
  const openContextMenu = useCallback((next: Selection, clientX: number, clientY: number) => {
    const alreadySelected = selections.some(
      (item) => selectionKey(item) === selectionKey(next),
    );
    const targets = alreadySelected ? selections : [next];
    if (!alreadySelected) setSelections(targets);
    setContextMenu({ anchor: next, targets, clientX, clientY });
    setNotice(`${targets.length} element${targets.length === 1 ? "" : "s"} ready for context actions`);
  }, [selections]);
  const beginGesture = useCallback((next: Selection): boolean => {
    if (isSelectionLocked(next)) {
      setNotice(`${next.collection.replaceAll("_", " ")} are locked · drag blocked`);
      return false;
    }
    dispatch({ type: "beginGesture", selection: next });
    return true;
  }, [isSelectionLocked]);
  const previewGesture = useCallback((next: Selection, changes: Record<string, unknown>): boolean => {
    const baseGraph = session.gesture?.before.graph ?? graph;
    if (next.collection === "rooms" && Array.isArray(changes.polygon)) {
      if (!baseGraph) return false;
      const plan = planRoomBoundaryTransform(
        baseGraph,
        next.id,
        changes.polygon as [number, number][],
      );
      if (!plan.valid) {
        setNotice(plan.reason ?? "Room boundary edit violates building relationships");
        return false;
      }
      const blockedDependency = plan.entries.find((entry) => isSelectionLocked(entry.selection));
      if (blockedDependency) {
        setNotice(`${blockedDependency.selection.id} is locked · room boundary edit blocked`);
        return false;
      }
      dispatch({
        type: "previewTransform",
        selection: next,
        entries: plan.entries,
        reason: "direct_room_boundary",
      });
      return true;
    }
    if (next.collection !== "walls" || !hasWallGeometryChange(changes)) {
      dispatch({ type: "previewGesture", selection: next, changes });
      return true;
    }
    if (!baseGraph) return false;
    const plan = planWallTransform(baseGraph, { [next.id]: changes });
    if (!plan.valid) {
      setNotice(plan.reason ?? "Wall edit violates building relationships");
      return false;
    }
    const blockedDependency = plan.entries.find((entry) => isSelectionLocked(entry.selection));
    if (blockedDependency) {
      setNotice(`${blockedDependency.selection.id} is locked · wall edit blocked`);
      return false;
    }
    dispatch({
      type: "previewTransform",
      selection: next,
      entries: plan.entries,
      reason: "direct_wall_manipulation",
    });
    return true;
  }, [dispatch, graph, isSelectionLocked, session.gesture?.before.graph]);
  const reviewNext = () => {
    if (!queue.length) {
      setNotice("Review queue is clear");
      return;
    }
    const current = selection ? queue.findIndex((item) => item.id === selection.id && item.collection === selection.collection) : -1;
    const next = queue[(current + 1) % queue.length];
    const nextEntity = graph ? findEntity(graph, next) : null;
    const nextLevelId = typeof nextEntity?.level_id === "string"
      ? nextEntity.level_id
      : typeof nextEntity?.from_level_id === "string"
        ? nextEntity.from_level_id
        : null;
    if (nextLevelId) setLevelId(nextLevelId);
    setHiddenCollections((currentHidden) => currentHidden.filter((item) => item !== next.collection));
    setHiddenEntities((currentHidden) => currentHidden.filter(
      (item) => selectionKey(item) !== selectionKey(next),
    ));
    setIsolatedEntities([]);
    setSelection(next);
    const priority = reviewPrioritiesByKey.get(selectionKey(next));
    setNotice(
      `${queue.length} items remain · ${priority?.percent ?? 0}/100 review risk · now reviewing ${next.id}`,
    );
  };

  const importFiles = async (files: FileList | null) => {
    if (!files) return;
    for (const file of Array.from(files)) {
      if (file.name.toLowerCase().endsWith(".json")) {
        try {
          const payload = JSON.parse(await file.text()) as PlanGraph;
          if (!Array.isArray(payload.walls) || !Array.isArray(payload.rooms)) throw new Error("not a PlanGraph");
          dispatch({ type: "load", graph: payload });
          setHiddenCollections([]);
          setLockedCollections([]);
          setHiddenEntities([]);
          setLockedEntities([]);
          setIsolatedEntities([]);
          setSelectionSets([]);
          setLevelId(payload.levels[0]?.id ?? "L1");
          setSelection(null);
          setNotice(`${file.name} loaded`);
        } catch (error) {
          setNotice(`Could not load ${file.name}: ${(error as Error).message}`);
        }
      } else if (file.type.startsWith("image/")) {
        if (sourceUrl.startsWith("blob:")) URL.revokeObjectURL(sourceUrl);
        setSourceUrl(URL.createObjectURL(file));
        setNotice(`${file.name} linked as source evidence`);
      }
    }
  };

  const exportPatch = async () => {
    if (!session.source || !session.present) return;
    const graphSha = await contentHash(session.source);
    downloadJson("dajoong-corrections.json", {
      schema_version: "buili.plan2bim-corrections.v1",
      expected_graph_sha256: graphSha,
      reviewer: "studio-user",
      operations: session.present.operations,
    });
    setNotice(`${session.present.operations.length} audited changes exported`);
  };

  const createWall = (from: [number, number], to: [number, number]) => {
    if (!graph) return;
    if (lockedCollections.includes("walls")) {
      setNotice("Walls are locked · unlock them in the Model Browser");
      return;
    }
    const id = uniqueId(graph.walls, `${levelId}:wall:manual`);
    const entity: WallEntity = {
      id,
      level_id: levelId,
      from,
      to,
      thickness_m: 0.12,
      height_m: graph.levels.find((item) => item.id === levelId)?.nominal_height_m ?? 3,
      wall_type: "interior",
      material: "gypsum",
      confidence: 1,
      uncertainty: 0,
      review_state: "accepted",
    };
    revealCollection("walls");
    dispatch({ type: "add", collection: "walls", entity });
    const nextSelection = { collection: "walls" as const, id };
    setIsolatedEntities((current) => current.length ? [...current, nextSelection] : current);
    setSelection(nextSelection);
    setNotice(`Wall placed · ${Math.hypot(to[0] - from[0], to[1] - from[1]).toFixed(3)} m`);
  };

  const addOpening = () => {
    if (!graph) return;
    if (lockedCollections.includes("openings")) {
      setNotice("Doors and windows are locked · unlock them in the Model Browser");
      return;
    }
    const wall = selection?.collection === "walls" ? graph.walls.find((item) => item.id === selection.id) : graph.walls.find((item) => item.level_id === levelId);
    if (!wall) return setNotice("Select a host wall first");
    const placement = findAvailableOpeningPlacement(wall, graph.openings, 0.9);
    if (!placement.valid || !placement.changes) {
      setNotice("No clear 900 mm span remains on this wall");
      return;
    }
    const id = uniqueId(graph.openings, `${levelId}:opening:manual`);
    const entity: OpeningEntity = {
      id,
      level_id: levelId,
      type: "door",
      wall_id: wall.id,
      center_m: placement.changes.center_m,
      x_m: placement.changes.x_m,
      width_m: placement.changes.width_m,
      height_m: 2.1,
      sill_height_m: 0,
      family_id: "generic-door",
      operation_type: "single_swing",
      handing: "start",
      swing_side: "positive",
      confidence: 1,
      uncertainty: 0,
      review_state: "accepted",
    };
    revealCollection("openings");
    dispatch({ type: "add", collection: "openings", entity });
    const nextSelection = { collection: "openings" as const, id };
    setIsolatedEntities((current) => current.length ? [...current, nextSelection] : current);
    setSelection(nextSelection);
    setNotice("Door placed in the largest clear host-wall span");
  };

  const addLevel = () => {
    if (!graph) return;
    let index = graph.levels.length + 1;
    while (graph.levels.some((level) => level.id === `L${index}`)) index += 1;
    const previousTop = graph.levels.reduce((top, level) => Math.max(
      top,
      Number(level.elevation_m ?? 0) + Number(level.nominal_height_m ?? 3),
    ), 0);
    const entity: LevelEntity = {
      id: `L${index}`,
      name: `Level ${index}`,
      elevation_m: previousTop,
      nominal_height_m: 3,
      confidence: 1,
      uncertainty: 0,
      review_state: "accepted",
      model_version: "human-correction",
    };
    dispatch({ type: "add", collection: "levels", entity });
    setLevelId(entity.id);
    setIsolatedEntities([]);
    setSelection(null);
    setNotice(`${entity.name} created at ${previousTop.toFixed(3)} m`);
  };

  const copySelectionToLevel = (targetLevelId: string) => {
    if (!graph || !selections.length) return;
    const result = copySelectionsToLevel(graph, selections, targetLevelId);
    const lockedTarget = result.items.find((item) => lockedCollections.includes(item.collection));
    if (lockedTarget) {
      setNotice(`Copy blocked · ${lockedTarget.collection.replaceAll("_", " ")} are locked`);
      return;
    }
    if (!result.items.length) {
      const conflict = result.conflicts[0];
      setNotice(
        conflict
          ? `Copy blocked · ${conflict.targetId} already occupies the target location`
          : result.warnings[0] ?? "The current selection cannot be copied to one level",
      );
      return;
    }
    const copiedCollections = new Set(result.items.map((item) => item.collection));
    setHiddenCollections((current) => current.filter((collection) => !copiedCollections.has(collection)));
    dispatch({ type: "addMany", items: result.items });
    setLevelId(targetLevelId);
    setIsolatedEntities(result.selections);
    setSelections(result.selections);
    const included = result.includedSourceSelections.length - selections.length;
    setNotice(
      `${result.items.length} BIM elements copied to ${targetLevelId}` +
      (included > 0 ? ` · ${included} hosted dependencies included` : ""),
    );
  };

  const editSelection = (next: Selection, changes: Record<string, unknown>) => {
    if (!graph) return;
    if (isSelectionLocked(next)) {
      setNotice(`${next.collection.replaceAll("_", " ")} are locked · edit blocked`);
      return;
    }
    if (next.collection === "walls" && hasWallGeometryChange(changes)) {
      const plan = planWallTransform(graph, { [next.id]: changes });
      if (!plan.valid) {
        setNotice(plan.reason ?? "Wall edit violates building relationships");
        return;
      }
      const blockedDependency = plan.entries.find((entry) => isSelectionLocked(entry.selection));
      if (blockedDependency) {
        setNotice(`${blockedDependency.selection.id} is locked · wall edit blocked`);
        return;
      }
      dispatch({
        type: "batchTransform",
        selections: plan.entries.map((entry) => entry.selection),
        changesById: Object.fromEntries(plan.entries.map((entry) => [entry.selection.id, entry.changes])),
        reason: "wall_property_edit",
      });
      setNotice(`Wall updated · ${plan.notices[0] ?? "host and room relationships verified"}`);
      return;
    }
    if (next.collection === "rooms" && Array.isArray(changes.polygon)) {
      const plan = planRoomBoundaryTransform(
        graph,
        next.id,
        changes.polygon as [number, number][],
      );
      if (!plan.valid) {
        setNotice(plan.reason ?? "Room boundary edit violates building relationships");
        return;
      }
      const blockedDependency = plan.entries.find((entry) => isSelectionLocked(entry.selection));
      if (blockedDependency) {
        setNotice(`${blockedDependency.selection.id} is locked · room boundary edit blocked`);
        return;
      }
      dispatch({
        type: "batchTransform",
        selections: plan.entries.map((entry) => entry.selection),
        changesById: Object.fromEntries(
          plan.entries.map((entry) => [entry.selection.id, entry.changes]),
        ),
        reason: "room_boundary_property_edit",
      });
      setNotice(`Room boundary updated${plan.notices.length ? ` · ${plan.notices[0]}` : " · contained objects verified"}`);
      return;
    }
    if (next.collection === "fixtures" && hasFixturePlacementChange(changes)) {
      const validation = validateFixtureEntityChanges(graph, next.id, changes);
      if (!validation.valid) {
        setNotice(validation.reason ?? "Component edit violates room or clearance constraints");
        return;
      }
      dispatch({
        type: "edit",
        selection: next,
        changes: validation.changesById[next.id] ?? changes,
      });
      setNotice(`Component updated${validation.notices.length ? ` · ${validation.notices[0]}` : " · room and clearance verified"}`);
      return;
    }
    if (next.collection !== "openings" || !hasOpeningGeometryChange(changes)) {
      dispatch({ type: "edit", selection: next, changes });
      return;
    }
    const opening = graph.openings.find((item) => item.id === next.id);
    if (!opening) return;
    const candidate = { ...opening, ...changes } as OpeningEntity;
    const host = graph.walls.find((item) => item.id === candidate.wall_id);
    if (!host) {
      setNotice("Opening edit blocked · host wall does not exist");
      return;
    }
    const placement = validateOpeningPlacement(candidate, host, graph.openings);
    if (!placement.valid || !placement.changes) {
      setNotice(openingPlacementNotice(placement.reason, placement.conflictId));
      return;
    }
    dispatch({
      type: "edit",
      selection: next,
      changes: { ...changes, ...placement.changes },
    });
  };

  const editSelections = (changes: Record<string, unknown>) => {
    if (!graph) return;
    const locked = selections.find(isSelectionLocked);
    if (locked) {
      setNotice(`${locked.collection.replaceAll("_", " ")} are locked · batch edit blocked`);
      return;
    }
    if (hasWallGeometryChange(changes) && selections.every((item) => item.collection === "walls")) {
      const proposed = Object.fromEntries(selections.map((item) => [item.id, changes]));
      const plan = planWallTransform(graph, proposed);
      if (!plan.valid) {
        setNotice(plan.reason ?? "Batch wall edit violates building relationships");
        return;
      }
      const blockedDependency = plan.entries.find((entry) => isSelectionLocked(entry.selection));
      if (blockedDependency) {
        setNotice(`${blockedDependency.selection.id} is locked · batch wall edit blocked`);
        return;
      }
      dispatch({
        type: "batchTransform",
        selections: plan.entries.map((entry) => entry.selection),
        changesById: Object.fromEntries(plan.entries.map((entry) => [entry.selection.id, entry.changes])),
        reason: "batch_wall_edit",
      });
      setNotice(`${selections.length} walls updated · hosted and room relationships verified`);
      return;
    }
    if (hasFixturePlacementChange(changes) && selections.every((item) => item.collection === "fixtures")) {
      const proposed = Object.fromEntries(selections.map((item) => [item.id, changes]));
      const validation = validateFixtureTransformChanges(graph, proposed);
      if (!validation.valid) {
        setNotice(validation.reason ?? "Batch component edit violates room or clearance constraints");
        return;
      }
      dispatch({
        type: "batchTransform",
        selections,
        changesById: validation.changesById,
        reason: "batch_fixture_edit",
      });
      setNotice(`${selections.length} components updated · room and clearance verified`);
      return;
    }
    if (!hasOpeningGeometryChange(changes) || !selections.every((item) => item.collection === "openings")) {
      dispatch({ type: "batchEdit", selections, changes });
      return;
    }
    const selectedIds = new Set(selections.map((item) => item.id));
    const candidates = graph.openings.map((opening) =>
      selectedIds.has(opening.id) ? ({ ...opening, ...changes } as OpeningEntity) : opening,
    );
    const changesById: EntityChanges = {};
    for (const selected of selections) {
      const candidate = candidates.find((item) => item.id === selected.id);
      if (!candidate) continue;
      const host = graph.walls.find((item) => item.id === candidate.wall_id);
      const placement = host
        ? validateOpeningPlacement(candidate, host, candidates)
        : { valid: false as const, reason: "invalid_wall" as const };
      if (!placement.valid || !placement.changes) {
        setNotice(openingPlacementNotice(placement.reason, placement.conflictId));
        return;
      }
      changesById[selected.id] = { ...changes, ...placement.changes };
    }
    dispatch({ type: "batchTransform", selections, changesById, reason: "batch_edit" });
  };

  const placeFamily = (family: FixtureFamily) => {
    if (!graph) return;
    if (lockedCollections.includes("fixtures")) {
      setNotice("Objects are locked · unlock them in the Model Browser");
      return;
    }
    const fixtureSelections = selections.filter((item) => item.collection === "fixtures");
    if (familyBrowserMode === "replace" && fixtureSelections.length) {
      const changesById = Object.fromEntries(fixtureSelections.map((item) => {
        const fixture = graph.fixtures.find((candidate) => candidate.id === item.id);
        const level = graph.levels.find((candidate) => candidate.id === fixture?.level_id);
        return [item.id, {
          type: family.type,
          family_id: family.id,
          discipline: family.discipline,
          size_m: family.size_m,
          material: family.material,
          mounting: family.mounting,
          base_elevation_m: defaultFixtureElevation(family, level),
        }];
      }));
      const validation = validateFixtureTransformChanges(graph, changesById);
      if (!validation.valid) {
        const message = validation.reason ?? "Replacement family does not fit the selected location";
        setFamilyBrowserError(message);
        setNotice(message);
        return;
      }
      dispatch({
        type: "batchTransform",
        selections: fixtureSelections,
        changesById: validation.changesById,
        reason: "replace_fixture_family",
      });
      setNotice(`${fixtureSelections.length} components replaced with ${family.name}`);
      setFamilyBrowserError("");
      setFamilyBrowserOpen(false);
      return;
    }
    revealCollection("fixtures");
    setPlacementFamily(family);
    setActiveTool("object");
    setViewMode("plan");
    setSelection(null);
    setNotice(`${family.name} ready · move over the plan, then click to place`);
    setFamilyBrowserError("");
    setFamilyBrowserOpen(false);
  };

  const placePendingFixture = useCallback((placement: FixturePlacement) => {
    if (!graph || !placementFamily || !placement.valid) return;
    if (lockedCollections.includes("fixtures")) {
      setNotice("Objects are locked · unlock them in the Model Browser");
      return;
    }
    const id = uniqueId(graph.fixtures, `${levelId}:fixture:manual`);
    const level = graph.levels.find((item) => item.id === levelId);
    const entity: FixtureEntity = {
      id,
      level_id: levelId,
      type: placementFamily.type,
      family_id: placementFamily.id,
      discipline: placementFamily.discipline,
      center_m: placement.center,
      size_m: placementFamily.size_m,
      yaw_deg: placement.yawDeg,
      base_elevation_m: defaultFixtureElevation(placementFamily, level),
      material: placementFamily.material,
      room_id: placement.roomId,
      host_wall_id: placement.hostWallId,
      mounting: placement.mounting,
      confidence: 1,
      uncertainty: 0,
      review_state: "accepted",
    };
    revealCollection("fixtures");
    dispatch({ type: "add", collection: "fixtures", entity });
    const nextSelection = { collection: "fixtures" as const, id };
    setIsolatedEntities((current) => current.length ? [...current, nextSelection] : current);
    setSelection(nextSelection);
    setNotice(`${placementFamily.name} placed in ${placement.roomId ?? levelId} · continue placing or press Esc`);
  }, [graph, levelId, lockedCollections, placementFamily, setSelection]);

  const cancelFixturePlacement = useCallback(() => {
    setPlacementFamily(null);
    setActiveTool("select");
    setNotice("Component placement finished");
  }, []);

  const browseFamilies = (mode: "insert" | "replace") => {
    if (lockedCollections.includes("fixtures")) {
      setNotice("Objects are locked · unlock them in the Model Browser");
      return;
    }
    setFamilyBrowserMode(mode);
    setFamilyBrowserError("");
    setFamilyBrowserOpen(true);
  };

  const selectionLocked = selections.some(isSelectionLocked);
  const contextTargets = contextMenu?.targets ?? [];
  const contextEntity = graph && contextMenu ? findEntity(graph, contextMenu.anchor) : null;
  const contextHidden = contextTargets.length > 0 && contextTargets.every(isSelectionHidden);
  const contextLocked = contextTargets.some(isSelectionLocked);
  const contextAllIndividuallyLocked = contextTargets.length > 0 && contextTargets.every(
    (item) => includesSelection(lockedEntities, item),
  );
  const contextInheritedLock = contextTargets.some((item) =>
    lockedCollections.includes(item.collection),
  );
  const contextAccepted = Boolean(
    graph && contextTargets.length > 0 && contextTargets.every((item) =>
      findEntity(graph, item)?.review_state === "accepted",
    ),
  );
  const contextCanDuplicate = contextTargets.length > 0 && contextTargets.every(
    (item) => item.collection === "fixtures" || item.collection === "vertical_connections",
  );
  const contextCanRehostOpening = contextTargets.length === 1
    && contextTargets[0].collection === "openings";
  const contextRelatedActions = graph && contextMenu && contextMenu.targets.length === 1
    ? relatedSelectionGroups(graph, contextMenu.anchor)
    : [];
  const toggleContextVisibility = () => {
    if (!contextTargets.length) return;
    const targetKeys = new Set(contextTargets.map(selectionKey));
    if (contextHidden) {
      const targetCollections = new Set(contextTargets.map((item) => item.collection));
      setHiddenCollections((current) => current.filter(
        (collection) => !targetCollections.has(collection),
      ));
      setHiddenEntities((current) => current.filter(
        (item) => !targetKeys.has(selectionKey(item)),
      ));
      setIsolatedEntities((current) => {
        if (!current.length) return current;
        const currentKeys = new Set(current.map(selectionKey));
        return [
          ...current,
          ...contextTargets.filter((item) => !currentKeys.has(selectionKey(item))),
        ];
      });
      setNotice(`${contextTargets.length} element${contextTargets.length === 1 ? "" : "s"} shown`);
      return;
    }
    setHiddenEntities((current) => {
      const currentKeys = new Set(current.map(selectionKey));
      return [
        ...current,
        ...contextTargets.filter((item) => !currentKeys.has(selectionKey(item))),
      ];
    });
    setNotice(`${contextTargets.length} element${contextTargets.length === 1 ? "" : "s"} hidden`);
  };
  const toggleContextLock = () => {
    if (!contextTargets.length || contextInheritedLock) return;
    const allLocked = contextTargets.every((item) => includesSelection(lockedEntities, item));
    const targetKeys = new Set(contextTargets.map(selectionKey));
    if (allLocked) {
      setLockedEntities((current) => current.filter(
        (item) => !targetKeys.has(selectionKey(item)),
      ));
      setNotice(`${contextTargets.length} element${contextTargets.length === 1 ? "" : "s"} unlocked`);
      return;
    }
    rawDispatch({ type: "cancelGesture" });
    setLockedEntities((current) => {
      const currentKeys = new Set(current.map(selectionKey));
      return [
        ...current,
        ...contextTargets.filter((item) => !currentKeys.has(selectionKey(item))),
      ];
    });
    setNotice(`${contextTargets.length} element${contextTargets.length === 1 ? "" : "s"} locked`);
  };
  const selectRelatedElements = (group: RelatedSelectionGroup) => {
    if (!graph || !group.selections.length) return;
    const targetKeys = new Set(group.selections.map(selectionKey));
    const targetCollections = new Set(group.selections.map((item) => item.collection));
    const firstEntity = findEntity(graph, group.selections[0]);
    const targetLevelId = typeof firstEntity?.level_id === "string"
      ? firstEntity.level_id
      : typeof firstEntity?.from_level_id === "string"
        ? firstEntity.from_level_id
        : null;
    if (targetLevelId) setLevelId(targetLevelId);
    setHiddenCollections((current) => current.filter(
      (collection) => !targetCollections.has(collection),
    ));
    setHiddenEntities((current) => current.filter(
      (item) => !targetKeys.has(selectionKey(item)),
    ));
    setIsolatedEntities((current) => {
      if (!current.length) return current;
      const currentKeys = new Set(current.map(selectionKey));
      return [
        ...current,
        ...group.selections.filter((item) => !currentKeys.has(selectionKey(item))),
      ];
    });
    setSelections(group.selections);
    setNotice(`${group.label} · ${group.selections.length} selected`);
  };
  const canArrange = !selectionLocked && selections.length >= 2 && selections.every(canArrangeSelection);
  const canDistribute = canArrange && selections.length >= 3;
  const duplicableSelections = selections.filter(
    (item) => item.collection === "fixtures" || item.collection === "vertical_connections",
  );
  const canDuplicate = !selectionLocked && duplicableSelections.length === selections.length && selections.length > 0;
  const canCopyToClipboard = selections.length > 0;
  const canPasteFromClipboard = Boolean(bimClipboard?.items.length);
  const canExactMove = !selectionLocked && selections.length > 0 && selections.every(canMoveExactly);
  const canExactRotate = !selectionLocked && selections.length > 0 && selections.every(canRotateExactly);
  const canRepeatPattern = !selectionLocked && selections.length > 0 && selections.every(canPatternSelection);
  const repeatPatternCenter = graph ? defaultMirrorCoordinates(graph, selections) : null;
  const exactRotationPivot = graph ? patternCenter(graph, selections) : null;
  const selectedOpening = selection?.collection === "openings"
    ? graph?.openings.find((item) => item.id === selection.id) ?? null
    : null;
  const canRehostOpening = !selectionLocked && selections.length === 1 && Boolean(selectedOpening);
  const canFlipDoor = !selectionLocked && selections.length === 1 && selectedOpening?.type === "door";
  const selectedWalls = selections
    .filter((item) => item.collection === "walls")
    .map((item) => graph?.walls.find((wall) => wall.id === item.id))
    .filter((item): item is WallEntity => item !== undefined);
  const canJoinWalls = !selectionLocked && selectedWalls.length === 2 && selections.length === 2 && selectedWalls[0].level_id === selectedWalls[1].level_id;

  const restoreRehostView = useCallback(() => {
    const previous = rehostViewStateRef.current;
    if (previous) {
      setHiddenCollections(previous.hiddenCollections);
      setHiddenEntities(previous.hiddenEntities);
      setIsolatedEntities(previous.isolatedEntities);
    }
    rehostViewStateRef.current = null;
    setRehostOpeningId(null);
  }, []);

  const startOpeningRehost = () => {
    if (!selection || !selectedOpening || !canRehostOpening) {
      setNotice("Select one unlocked door or window before choosing a new host");
      return;
    }
    if (!rehostViewStateRef.current) {
      rehostViewStateRef.current = {
        hiddenCollections,
        hiddenEntities,
        isolatedEntities,
      };
    }
    setPlacementFamily(null);
    setActiveTool("select");
    setViewMode("plan");
    setHiddenCollections((current) => current.filter(
      (collection) => collection !== "walls" && collection !== "openings",
    ));
    setHiddenEntities((current) => current.filter(
      (item) => item.collection !== "walls" && item.collection !== "openings",
    ));
    setIsolatedEntities([]);
    setRehostOpeningId(selectedOpening.id);
    setNotice("Pick a highlighted host wall · click near the preferred opening position");
  };

  const cancelOpeningRehost = useCallback(() => {
    restoreRehostView();
    setNotice("Opening rehost cancelled");
  }, [restoreRehostView]);

  const pickOpeningHost = useCallback((targetWallId: string, point: [number, number]) => {
    if (!graph || !rehostOpeningId) return;
    const targetSelection: Selection = { collection: "walls", id: targetWallId };
    if (isSelectionLocked(targetSelection)) {
      setNotice(`${targetWallId} is locked · choose an unlocked host wall`);
      return;
    }
    const plan = planOpeningRehost(graph, rehostOpeningId, targetWallId, point);
    if (!plan.valid) {
      setNotice(plan.reason ?? "The selected wall cannot host this opening");
      return;
    }
    dispatch({
      type: "batchTransform",
      selections: [plan.selection],
      changesById: { [plan.selection.id]: plan.changes },
      reason: "opening_rehost",
    });
    restoreRehostView();
    setNotice(`Opening rehosted${plan.notices.length ? ` · ${plan.notices.join(" · ")}` : ""}`);
  }, [dispatch, graph, isSelectionLocked, rehostOpeningId, restoreRehostView]);

  const flipDoorHanding = () => {
    if (!selection || !selectedOpening || !canFlipDoor) return;
    dispatch({
      type: "batchTransform",
      selections: [selection],
      changesById: { [selection.id]: { handing: toggleDoorHanding(selectedOpening.handing) } },
      reason: "flip_door_handing",
    });
    setNotice("Door hinge flipped in the host wall frame");
  };

  const reverseDoorSwing = () => {
    if (!selection || !selectedOpening || !canFlipDoor) return;
    dispatch({
      type: "batchTransform",
      selections: [selection],
      changesById: { [selection.id]: { swing_side: toggleDoorSwingSide(selectedOpening.swing_side) } },
      reason: "reverse_door_swing",
    });
    setNotice("Door swing side reversed");
  };

  const joinSelectedWalls = () => {
    if (!canJoinWalls) return;
    const changesById = joinWallEndpointChanges(selectedWalls);
    if (!Object.keys(changesById).length) return;
    if (applyWallConstraint(changesById, "join_wall_endpoints")) {
      setNotice("Nearest wall endpoints joined · room, opening, and host relationships verified");
    }
  };

  const cornerSelectedWalls = () => {
    if (!canJoinWalls) return;
    const changesById = cornerWallChanges(selectedWalls);
    if (!Object.keys(changesById).length) {
      setNotice("Selected wall axes are parallel or too far apart");
      return;
    }
    if (applyWallConstraint(changesById, "trim_extend_wall_corner")) {
      setNotice("Walls trimmed or extended to an exact corner · BIM relationships verified");
    }
  };

  const applyWallConstraint = (changesById: EntityChanges, reason: string): boolean => {
    if (!graph || !canJoinWalls) return false;
    const plan = planWallTransform(graph, changesById);
    if (!plan.valid) {
      setNotice(plan.reason ?? "Wall constraint violates building relationships");
      return false;
    }
    const blockedDependency = plan.entries.find((entry) => isSelectionLocked(entry.selection));
    if (blockedDependency) {
      setNotice(`${blockedDependency.selection.id} is locked · wall constraint blocked`);
      return false;
    }
    const plannedSelections = plan.entries.map((entry) => entry.selection);
    const plannedChanges = Object.fromEntries(
      plan.entries.map((entry) => [entry.selection.id, entry.changes]),
    );
    const references = selectedWalls.map((wall) => {
      const changes = changesById[wall.id] ?? {};
      const handle: "from" | "to" = Object.prototype.hasOwnProperty.call(changes, "from")
        ? "from"
        : "to";
      return { collection: "walls" as const, entity_id: wall.id, handle };
    });
    const key = references
      .map((reference) => `${reference.entity_id}:${reference.handle}`)
      .sort()
      .join("|");
    const existing = (graph.constraints ?? []).find(
      (constraint) => constraint.type === "coincident" && constraint.references
        .map((reference) => `${reference.entity_id}:${reference.handle}`)
        .sort()
        .join("|") === key,
    );
    if (existing) {
      dispatch({
        type: "batchTransform",
        selections: plannedSelections,
        changesById: plannedChanges,
        reason,
      });
      return true;
    }
    const constraint: GeometricConstraintEntity = {
      id: uniqueId(graph.constraints ?? [], `${levelId}:constraint:coincident`),
      level_id: levelId,
      type: "coincident",
      references,
      confidence: 1,
      uncertainty: 0,
      review_state: "accepted",
      model_version: "human-correction",
    };
    dispatch({
      type: "constrainWalls",
      selections: plannedSelections,
      changesById: plannedChanges,
      constraint,
      reason,
    });
    return true;
  };

  const transformSelection = (
    changesById: EntityChanges,
    reason: string,
    message: string,
  ): boolean => {
    if (selectionLocked) {
      setNotice("Selection is locked · transform blocked");
      return false;
    }
    if (!Object.keys(changesById).length) return false;
    const prepared = graph
      ? prepareBimTransform(graph, selections, changesById)
      : { valid: true as const, selections, changesById, notices: [] as string[] };
    if (!prepared.valid) {
      setNotice(prepared.reason ?? "Transform violates BIM relationships or clearance constraints");
      return false;
    }
    const blockedDependency = prepared.selections.find(isSelectionLocked);
    if (blockedDependency) {
      setNotice(`${blockedDependency.id} is locked · dependent BIM transform blocked`);
      return false;
    }
    dispatch({
      type: "batchTransform",
      selections: prepared.selections,
      changesById: prepared.changesById,
      reason,
    });
    setNotice(`${message}${prepared.notices.length ? ` · ${prepared.notices[0]}` : ""}`);
    return true;
  };

  const applyExactTranslation = (
    delta: [number, number],
    reason = "exact_translation",
  ): string | null => {
    if (!graph) return "No editable graph is loaded.";
    if (!canExactMove) return "The current selection cannot be moved as one exact transform.";
    const result = exactTranslationChanges(graph, selections, delta);
    if (!result.valid) return result.reason ?? "The exact move violates model constraints.";
    const prepared = prepareBimTransform(graph, selections, result.changesById);
    if (!prepared.valid) {
      return prepared.reason ?? "The exact move violates BIM relationships or clearance constraints.";
    }
    const distance = Math.hypot(delta[0], delta[1]);
    const committed = transformSelection(
      result.changesById,
      reason,
      `${selections.length} element${selections.length === 1 ? "" : "s"} moved ${distance.toFixed(3)} m${result.notices.length ? ` · ${result.notices[0]}` : ""}`,
    );
    if (!committed) return "The exact move violates component room or clearance constraints.";
    setExactMoveOpen(false);
    return null;
  };

  const applyExactRotation = (
    deltaDegrees: number,
    pivot: [number, number],
  ): string | null => {
    if (!graph || !canExactRotate) return "Select unlocked placed components to rotate.";
    const result = exactRotationChanges(graph, selections, pivot, deltaDegrees);
    if (!result.valid) return result.reason ?? "The exact rotation violates model constraints.";
    const prepared = prepareBimTransform(graph, selections, result.changesById);
    if (!prepared.valid) {
      return prepared.reason ?? "The exact rotation violates BIM relationships or clearance constraints.";
    }
    if (!transformSelection(
      result.changesById,
      "exact_group_rotation",
      `${selections.length} component${selections.length === 1 ? "" : "s"} rotated ${deltaDegrees.toFixed(1)}° about (${pivot[0].toFixed(3)}, ${pivot[1].toFixed(3)}) · one Undo step`,
    )) return "The exact rotation violates component room or clearance constraints.";
    setExactRotateOpen(false);
    return null;
  };

  const commitModelTransform = useCallback((
    items: Selection[],
    transform: ModelTransformCommit,
  ): boolean => {
    if (!graph) return false;
    const locked = items.find(isSelectionLocked);
    if (locked) {
      setNotice(`${locked.id} is locked · 3D transform blocked`);
      return false;
    }
    if (transform.delta_m) {
      const result = exactTranslationChanges(graph, items, transform.delta_m);
      if (!result.valid) {
        setNotice(result.reason ?? "The 3D move violates model constraints.");
        return false;
      }
      const prepared = prepareBimTransform(graph, items, result.changesById);
      if (!prepared.valid) {
        setNotice(prepared.reason ?? "The 3D move violates BIM relationships or clearance constraints.");
        return false;
      }
      const blockedDependency = prepared.selections.find(isSelectionLocked);
      if (blockedDependency) {
        setNotice(`${blockedDependency.id} is locked · dependent 3D transform blocked`);
        return false;
      }
      dispatch({
        type: "batchTransform",
        selections: prepared.selections,
        changesById: prepared.changesById,
        reason: "3d_gizmo_translation",
      });
      const distance = Math.hypot(...transform.delta_m);
      setNotice(`${items.length} element${items.length === 1 ? "" : "s"} moved ${distance.toFixed(3)} m in 3D · one Undo step${prepared.notices.length ? ` · ${prepared.notices[0]}` : ""}`);
      return true;
    }
    if (typeof transform.rotation_delta_deg === "number" && transform.pivot_m) {
      const result = exactRotationChanges(
        graph,
        items,
        transform.pivot_m,
        transform.rotation_delta_deg,
      );
      if (!result.valid) {
        setNotice(result.reason ?? "The 3D rotation violates model constraints.");
        return false;
      }
      const prepared = prepareBimTransform(graph, items, result.changesById);
      if (!prepared.valid) {
        setNotice(prepared.reason ?? "The 3D rotation violates BIM relationships or clearance constraints.");
        return false;
      }
      const blockedDependency = prepared.selections.find(isSelectionLocked);
      if (blockedDependency) {
        setNotice(`${blockedDependency.id} is locked · dependent 3D transform blocked`);
        return false;
      }
      dispatch({
        type: "batchTransform",
        selections: prepared.selections,
        changesById: prepared.changesById,
        reason: "3d_gizmo_rotation",
      });
      setNotice(`${items.length} component${items.length === 1 ? "" : "s"} rotated ${transform.rotation_delta_deg.toFixed(1)}° about the shared pivot · one Undo step${prepared.notices.length ? ` · ${prepared.notices[0]}` : ""}`);
      return true;
    }
    return false;
  }, [dispatch, graph, isSelectionLocked]);

  const alignSelection = (mode: AlignmentMode) => {
    if (!graph) return;
    const plan = planArrangement(graph, selections, { type: "align", mode });
    if (!plan.valid) {
      setNotice(plan.reason ?? "The selected components cannot be aligned safely");
      return;
    }
    transformSelection(
      plan.changesById,
      `align_${mode}`,
      `${plan.selections.length} component${plan.selections.length === 1 ? "" : "s"} aligned to ${plan.keySelection?.id ?? "the key object"} · one Undo step${plan.notices.length ? ` · ${plan.notices[0]}` : ""}`,
    );
  };

  const distributeSelection = (axis: DistributionAxis) => {
    if (!graph) return;
    const plan = planArrangement(graph, selections, { type: "distribute", axis });
    if (!plan.valid) {
      setNotice(plan.reason ?? "The selected components cannot be distributed safely");
      return;
    }
    transformSelection(
      plan.changesById,
      `distribute_${axis}`,
      `${selections.length} components spaced equally ${axis === "horizontal" ? "horizontally" : "vertically"} · outer anchors fixed · one Undo step${plan.notices.length ? ` · ${plan.notices[0]}` : ""}`,
    );
  };

  const nudgeSelection = (delta: [number, number]) => {
    const error = applyExactTranslation(delta, "keyboard_nudge");
    if (error) setNotice(error);
  };

  const copySelectionToClipboard = () => {
    if (!graph || !canCopyToClipboard) {
      setNotice("Select BIM elements before copying");
      return;
    }
    const bundle = createBimClipboardBundle(graph, selections);
    if (!bundle.items.length) {
      setNotice(bundle.warnings[0] ?? "The current selection cannot be copied as BIM geometry");
      return;
    }
    setBimClipboard(bundle);
    const dependencies = bundle.included_selections.length - bundle.source_selections.length;
    setNotice(
      `${bundle.items.length} BIM element${bundle.items.length === 1 ? "" : "s"} copied` +
      (dependencies > 0 ? ` · ${dependencies} hosted dependenc${dependencies === 1 ? "y" : "ies"} included` : "") +
      " · paste with Ctrl+V",
    );
  };

  const pasteBimClipboard = () => {
    if (!graph || !bimClipboard?.items.length) {
      setNotice("The BIM clipboard is empty");
      return;
    }
    const blockedCollection = bimClipboard.items.find((item) => lockedCollections.includes(item.collection));
    if (blockedCollection) {
      setNotice(`${blockedCollection.collection.replaceAll("_", " ")} are locked · paste blocked`);
      return;
    }
    const plan = planBimPaste(graph, bimClipboard, levelId, snapIncrementM || 0.05);
    if (!plan.valid) {
      setNotice(plan.reason ?? "No collision-free BIM paste location was found");
      return;
    }
    const pastedCollections = new Set(plan.items.map((item) => item.collection));
    setHiddenCollections((current) => current.filter((collection) => !pastedCollections.has(collection)));
    dispatch({ type: "addMany", items: plan.items, reason: "paste_bim_clipboard" });
    setSelections(plan.selections);
    setIsolatedEntities((current) => current.length ? [...current, ...plan.selections] : current);
    const offset = plan.offset_m ?? [0, 0];
    const distance = Math.hypot(offset[0], offset[1]);
    setNotice(
      `${plan.items.length} BIM element${plan.items.length === 1 ? "" : "s"} pasted` +
      (distance > 0.0001 ? ` · nearest clear offset ${distance.toFixed(3)} m` : " · pasted in place on this level") +
      " · relationships remapped · one Undo step",
    );
  };

  const duplicateSelection = () => {
    if (!graph || !canDuplicate) return;
    const items: Array<{ collection: CollectionName; entity: BaseEntity }> = [];
    for (const selected of duplicableSelections) {
      const entity = findEntity(graph, selected);
      if (!entity) continue;
      const existing = [
        ...((graph[selected.collection] as BaseEntity[]) ?? []),
        ...items
          .filter((item) => item.collection === selected.collection)
          .map((item) => item.entity),
      ];
      const clone = structuredClone(entity);
      clone.id = uniqueId(existing, `${entity.id}:copy`);
      clone.confidence = 1;
      clone.uncertainty = 0;
      clone.review_state = "accepted";
      clone.model_version = "human-correction";
      items.push({ collection: selected.collection, entity: clone });
    }
    if (!items.length) return;
    const fixtureClones = items
      .filter((item) => item.collection === "fixtures")
      .map((item) => item.entity as FixtureEntity);
    const fixtureCopy = fixtureClones.length
      ? findNearestValidFixtureCopy(graph, fixtureClones, snapIncrementM || 0.05)
      : null;
    if (fixtureCopy && !fixtureCopy.valid) {
      setNotice(fixtureCopy.reason ?? "No clear duplicate placement was found");
      return;
    }
    const offset = fixtureCopy?.offset ?? [snapIncrementM || 0.05, snapIncrementM || 0.05] as [number, number];
    const normalizedFixtures = new Map((fixtureCopy?.fixtures ?? []).map((fixture) => [fixture.id, fixture]));
    for (const item of items) {
      if (item.collection === "fixtures") {
        item.entity = normalizedFixtures.get(item.entity.id) ?? item.entity;
        continue;
      }
      const center = item.entity.center_m as [number, number];
      item.entity.center_m = [Number(center[0]) + offset[0], Number(center[1]) + offset[1]];
    }
    dispatch({ type: "addMany", items });
    const duplicatedSelections = items.map((item) => ({ collection: item.collection, id: item.entity.id }));
    setIsolatedEntities((current) => current.length ? [...current, ...duplicatedSelections] : current);
    setSelections(duplicatedSelections);
    setNotice(`${items.length} component${items.length === 1 ? "" : "s"} duplicated to the nearest clear location`);
  };

  const applyMirrorPattern = (
    axis: MirrorAxis,
    coordinateM: number,
    keepOriginal: boolean,
  ): string | null => {
    if (!graph || !canRepeatPattern) return "Select unlocked placed components to mirror.";
    const result = mirrorPattern(graph, selections, axis, coordinateM, keepOriginal);
    if (!result.valid) return result.reason ?? "Mirror could not be created.";
    if (result.items.length) {
      const fixtureItems = result.items
        .filter((item) => item.collection === "fixtures")
        .map((item) => item.entity as FixtureEntity);
      const validation = validateNewFixtures(graph, fixtureItems);
      if (!validation.valid) return validation.reason ?? "Mirrored components violate room or clearance constraints.";
      const normalized = new Map(validation.fixtures.map((fixture) => [fixture.id, fixture]));
      const items = result.items.map((item) => item.collection === "fixtures"
        ? { ...item, entity: normalized.get(item.entity.id) ?? item.entity }
        : item);
      dispatch({ type: "addMany", items, reason: "mirror_component" });
      const nextSelections = items.map((item) => ({ collection: item.collection, id: item.entity.id }));
      setSelections(nextSelections);
      setIsolatedEntities((current) => current.length ? [...current, ...nextSelections] : current);
    } else {
      const validation = validateFixtureTransformChanges(graph, result.changesById);
      if (!validation.valid) return validation.reason ?? "Mirrored components violate room or clearance constraints.";
      dispatch({
        type: "batchTransform",
        selections,
        changesById: validation.changesById,
        reason: "mirror_component",
      });
    }
    setPatternOpen(null);
    setNotice(`${selections.length} component${selections.length === 1 ? "" : "s"} mirrored about ${axis === "vertical" ? "X" : "Y"} = ${coordinateM.toFixed(3)} m`);
    return null;
  };

  const applyLinearArray = (count: number, step: [number, number]): string | null => {
    if (!graph || !canRepeatPattern) return "Select unlocked placed components to array.";
    const result = linearArrayPattern(graph, selections, count, step);
    if (!result.valid) return result.reason ?? "Linear array could not be created.";
    const fixtureItems = result.items
      .filter((item) => item.collection === "fixtures")
      .map((item) => item.entity as FixtureEntity);
    const validation = validateNewFixtures(graph, fixtureItems);
    if (!validation.valid) return validation.reason ?? "Array components violate room or clearance constraints.";
    const normalized = new Map(validation.fixtures.map((fixture) => [fixture.id, fixture]));
    const items = result.items.map((item) => item.collection === "fixtures"
      ? { ...item, entity: normalized.get(item.entity.id) ?? item.entity }
      : item);
    dispatch({ type: "addMany", items, reason: "linear_array_component" });
    const finalInstance = items.slice(-selections.length).map((item) => ({
      collection: item.collection,
      id: item.entity.id,
    }));
    setSelections(finalInstance);
    setIsolatedEntities((current) => current.length ? [...current, ...items.map((item) => ({ collection: item.collection, id: item.entity.id }))] : current);
    setPatternOpen(null);
    setNotice(`${count} total instances created · ${items.length} new components · one Undo step`);
    return null;
  };

  const deleteSelected = () => {
    if (!graph || !selections.length) return;
    if (selectionLocked) {
      setNotice("Selection is locked · delete blocked");
      return;
    }
    const selectedIds = new Set(selections.map((item) => item.id));
    const blockedWall = selections.some(
      (item) =>
        item.collection === "walls" &&
        graph.openings.some(
          (opening) => opening.wall_id === item.id && !selectedIds.has(opening.id),
        ),
    );
    const blockedRoom = selections.some(
      (item) =>
        item.collection === "rooms" &&
        graph.fixtures.some(
          (fixture) => fixture.room_id === item.id && !selectedIds.has(fixture.id),
        ),
    );
    const blockedConstraint = selections.some(
      (item) =>
        item.collection === "walls" &&
        (graph.constraints ?? []).some(
          (constraint) =>
            constraint.references.some((reference) => reference.entity_id === item.id) &&
            !selectedIds.has(constraint.id),
        ),
    );
    if (blockedWall || blockedRoom || blockedConstraint) {
      setNotice("Selection contains hosted elements · include or rehost them first");
      return;
    }
    if (selections.length === 1) {
      dispatch({ type: "delete", selection: selections[0] });
    } else {
      dispatch({ type: "batchDelete", selections });
    }
    setSelection(null);
  };

  const acceptSelected = () => {
    if (!selections.length) return;
    if (selectionLocked) {
      setNotice("Selection is locked · review state unchanged");
      return;
    }
    dispatch(
      selections.length === 1
        ? { type: "accept", selection: selections[0] }
        : { type: "batchAccept", selections },
    );
    setNotice(`${selections.length} element${selections.length === 1 ? "" : "s"} accepted`);
  };

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const isTyping =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.tagName === "SELECT" ||
        target?.isContentEditable;
      if (isTyping) return;
      if (!event.ctrlKey && !event.metaKey && event.key.toLowerCase() === "i") {
        if (!selections.length && !isolatedEntities.length) return;
        event.preventDefault();
        if (isolatedEntities.length) exitIsolation();
        else isolateSelection();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "c") {
        if (!canCopyToClipboard) return;
        event.preventDefault();
        copySelectionToClipboard();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "v") {
        if (!canPasteFromClipboard) return;
        event.preventDefault();
        pasteBimClipboard();
        return;
      }
      if (!selections.length) return;
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "d") {
        if (!canDuplicate) return;
        event.preventDefault();
        duplicateSelection();
        return;
      }
      if (!event.ctrlKey && !event.metaKey && event.key.toLowerCase() === "m") {
        if (event.shiftKey && canRepeatPattern) {
          event.preventDefault();
          setPatternOpen("mirror");
          return;
        }
        if (!canExactMove) return;
        event.preventDefault();
        setExactMoveOpen(true);
        return;
      }
      if (!event.ctrlKey && !event.metaKey && event.shiftKey && event.key.toLowerCase() === "a") {
        if (!canRepeatPattern) return;
        event.preventDefault();
        setPatternOpen("array");
        return;
      }
      if (!event.ctrlKey && !event.metaKey && event.key.toLowerCase() === "r") {
        if (!canExactRotate) return;
        event.preventDefault();
        setExactRotateOpen(true);
        return;
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        deleteSelected();
        return;
      }
      if (canRehostOpening && event.shiftKey && event.key.toLowerCase() === "h") {
        event.preventDefault();
        startOpeningRehost();
        return;
      }
      if (canFlipDoor && !event.shiftKey && event.key.toLowerCase() === "h") {
        event.preventDefault();
        flipDoorHanding();
        return;
      }
      if (canFlipDoor && event.key.toLowerCase() === "s") {
        event.preventDefault();
        reverseDoorSwing();
        return;
      }
      if (event.altKey && !event.ctrlKey && !event.metaKey && ["x", "y"].includes(event.key.toLowerCase())) {
        const horizontal = event.key.toLowerCase() === "x";
        if (event.shiftKey) {
          if (!canDistribute) return;
          event.preventDefault();
          distributeSelection(horizontal ? "horizontal" : "vertical");
        } else {
          if (!canArrange) return;
          event.preventDefault();
          alignSelection(horizontal ? "center-x" : "center-y");
        }
        return;
      }
      const deltas: Partial<Record<string, [number, number]>> = {
        ArrowLeft: [-1, 0],
        ArrowRight: [1, 0],
        ArrowUp: [0, -1],
        ArrowDown: [0, 1],
      };
      const direction = deltas[event.key];
      if (!direction || !canExactMove) return;
      event.preventDefault();
      const base = event.altKey ? 0.01 : (snapIncrementM || 0.05);
      const multiplier = event.shiftKey ? 10 : 1;
      nudgeSelection([
        direction[0] * base * multiplier,
        direction[1] * base * multiplier,
      ]);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const studioCommands: StudioCommand[] = [
    {
      id: "convert-drawing",
      label: "Convert a drawing",
      group: "Import",
      aliases: ["upload PDF", "image to BIM", "plan to model"],
      run: () => setConversionOpen(true),
    },
    {
      id: "add-wall",
      label: "Add wall",
      group: "Model",
      aliases: ["draw wall", "partition"],
      enabled: !lockedCollections.includes("walls"),
      disabledReason: "Walls are locked in the Model Browser",
      run: () => setActiveTool("wall"),
    },
    {
      id: "add-opening",
      label: "Add opening",
      group: "Model",
      aliases: ["door", "window", "place opening"],
      enabled: !lockedCollections.includes("openings") && Boolean(graph?.walls.some((wall) => wall.level_id === levelId)),
      disabledReason: lockedCollections.includes("openings") ? "Openings are locked" : "Add a wall on this level first",
      run: addOpening,
    },
    {
      id: "add-object",
      label: "Add object",
      group: "Model",
      aliases: ["component", "fixture", "equipment", "family"],
      enabled: !lockedCollections.includes("fixtures"),
      disabledReason: "Objects are locked in the Model Browser",
      run: () => browseFamilies("insert"),
    },
    {
      id: "add-level",
      label: "Add building level",
      group: "Model",
      aliases: ["floor", "story", "storey"],
      run: addLevel,
    },
    {
      id: "measure-distance",
      label: "Measure distance",
      group: "Inspect",
      aliases: ["dimension", "ruler", "length"],
      enabled: !lockedCollections.includes("dimensions"),
      disabledReason: "Dimensions are locked in the Model Browser",
      run: () => setActiveTool("measure"),
    },
    {
      id: "review-next",
      label: "Review next item",
      group: "Review",
      aliases: ["risk", "confidence", "triage"],
      enabled: queue.length > 0,
      disabledReason: "The guided review queue is clear",
      run: reviewNext,
    },
    {
      id: "model-assurance",
      label: "Open model assurance",
      group: "Review",
      aliases: ["quality", "verify", "benchmark", "validation"],
      run: () => setQualityOpen(true),
    },
    {
      id: "copy-bim",
      label: "Copy BIM selection with hosted dependencies",
      group: "Clipboard",
      aliases: ["copy component", "copy wall"],
      shortcut: "Ctrl C",
      enabled: canCopyToClipboard,
      disabledReason: "Select at least one BIM element",
      run: copySelectionToClipboard,
    },
    {
      id: "paste-bim",
      label: "Paste BIM selection at nearest clear location",
      group: "Clipboard",
      aliases: ["paste component", "paste wall"],
      shortcut: "Ctrl V",
      enabled: canPasteFromClipboard,
      disabledReason: "Copy a compatible BIM selection first",
      run: pasteBimClipboard,
    },
    {
      id: "duplicate",
      label: "Duplicate selected components",
      group: "Modify",
      aliases: ["copy in place", "clone"],
      shortcut: "Ctrl D",
      enabled: canDuplicate,
      disabledReason: "The current selection cannot be duplicated",
      run: duplicateSelection,
    },
    {
      id: "move-exact",
      label: "Move selected elements by exact offset",
      group: "Modify",
      aliases: ["translate", "precision move", "offset"],
      shortcut: "M",
      enabled: canExactMove,
      disabledReason: "Select movable unlocked elements",
      run: () => setExactMoveOpen(true),
    },
    {
      id: "rotate-exact",
      label: "Rotate selected components by exact angle",
      group: "Modify",
      aliases: ["precision rotate", "angle"],
      shortcut: "R",
      enabled: canExactRotate,
      disabledReason: "Select rotatable unlocked components",
      run: () => setExactRotateOpen(true),
    },
    {
      id: "mirror",
      label: "Mirror selected components",
      group: "Modify",
      aliases: ["reflect", "flip geometry"],
      shortcut: "Shift M",
      enabled: canRepeatPattern,
      disabledReason: "Select compatible unlocked components",
      run: () => setPatternOpen("mirror"),
    },
    {
      id: "linear-array",
      label: "Create linear component array",
      group: "Modify",
      aliases: ["repeat", "pattern", "multiple copies"],
      shortcut: "Shift A",
      enabled: canRepeatPattern,
      disabledReason: "Select compatible unlocked components",
      run: () => setPatternOpen("array"),
    },
    ...([
      ["align-left", "Align left edges to key object", "left"],
      ["align-center-x", "Align horizontal centers to key object", "center-x"],
      ["align-right", "Align right edges to key object", "right"],
      ["align-top", "Align top edges to key object", "top"],
      ["align-center-y", "Align vertical centers to key object", "center-y"],
      ["align-bottom", "Align bottom edges to key object", "bottom"],
    ] as const).map(([id, label, mode]) => ({
      id,
      label,
      group: "Arrange",
      aliases: ["key object", "line up"],
      enabled: canArrange,
      disabledReason: "Select two compatible unlocked elements",
      run: () => alignSelection(mode),
    })),
    {
      id: "distribute-horizontal",
      label: "Space selected components equally horizontally",
      group: "Arrange",
      aliases: ["distribute x", "equal gap"],
      shortcut: "Shift Alt X",
      enabled: canDistribute,
      disabledReason: "Select three compatible unlocked elements",
      run: () => distributeSelection("horizontal"),
    },
    {
      id: "distribute-vertical",
      label: "Space selected components equally vertically",
      group: "Arrange",
      aliases: ["distribute y", "equal gap"],
      shortcut: "Shift Alt Y",
      enabled: canDistribute,
      disabledReason: "Select three compatible unlocked elements",
      run: () => distributeSelection("vertical"),
    },
    {
      id: "flip-door-hinge",
      label: "Flip selected door hinge",
      group: "Modify",
      aliases: ["door handing", "reverse hinge"],
      shortcut: "H",
      enabled: canFlipDoor,
      disabledReason: "Select one unlocked door",
      run: flipDoorHanding,
    },
    {
      id: "reverse-door-swing",
      label: "Reverse selected door swing",
      group: "Modify",
      aliases: ["swing side", "flip door"],
      shortcut: "S",
      enabled: canFlipDoor,
      disabledReason: "Select one unlocked door",
      run: reverseDoorSwing,
    },
    {
      id: "rehost-opening",
      label: "Pick new host wall for selected opening",
      group: "Modify",
      aliases: ["rehost door", "rehost window", "change wall"],
      shortcut: "Shift H",
      enabled: canRehostOpening,
      disabledReason: "Select one unlocked door or window",
      run: startOpeningRehost,
    },
    {
      id: "join-walls",
      label: "Join selected wall endpoints",
      group: "Modify",
      aliases: ["connect wall", "coincident"],
      enabled: canJoinWalls,
      disabledReason: "Select two unlocked walls on the same level",
      run: joinSelectedWalls,
    },
    {
      id: "corner-walls",
      label: "Trim or extend selected walls to corner",
      group: "Modify",
      aliases: ["wall corner", "intersect", "trim extend"],
      enabled: canJoinWalls,
      disabledReason: "Select two unlocked walls on the same level",
      run: cornerSelectedWalls,
    },
    { id: "view-split", label: "Show split view", group: "View", aliases: ["2D and 3D"], run: () => setViewMode("split") },
    { id: "view-plan", label: "Show 2D plan", group: "View", aliases: ["drawing", "floor plan"], run: () => setViewMode("plan") },
    { id: "view-model", label: "Show 3D model", group: "View", aliases: ["BIM", "perspective"], run: () => setViewMode("model") },
    {
      id: "toggle-history",
      label: "Toggle edit history timeline",
      group: "History",
      aliases: ["undo timeline", "design history"],
      run: toggleHistoryCollapsed,
    },
    {
      id: "history-previous",
      label: "Go to previous edit history state",
      group: "History",
      aliases: ["undo state", "back"],
      enabled: session.past.length > 0,
      disabledReason: "Already at the imported state",
      run: () => jumpToHistory(session.past.length - 1),
    },
    {
      id: "history-next",
      label: "Go to next edit history state",
      group: "History",
      aliases: ["redo state", "forward"],
      enabled: session.future.length > 0,
      disabledReason: "Already at the latest edit",
      run: () => jumpToHistory(session.past.length + 1),
    },
    {
      id: "isolate-selection",
      label: "Isolate selected elements",
      group: "View",
      aliases: ["hide others", "focus"],
      shortcut: "I",
      enabled: selections.length > 0,
      disabledReason: "Select one or more elements",
      run: isolateSelection,
    },
    {
      id: "save-selection-set",
      label: "Save current selection as a set",
      group: "Selection Sets",
      aliases: ["named selection", "save selection", "workset"],
      enabled: selections.length > 0,
      disabledReason: "Select one or more model elements",
      run: () => saveCurrentSelectionSet(),
    },
    ...selectionSets.flatMap((set) => [
      {
        id: `recall-${set.id}`,
        label: `Select set: ${set.name}`,
        group: "Selection Sets",
        aliases: [set.name, "named selection", "recall set"],
        run: () => recallSelectionSet(set),
      },
      {
        id: `isolate-${set.id}`,
        label: `Isolate set: ${set.name}`,
        group: "Selection Sets",
        aliases: [set.name, "named isolation", "focus set"],
        run: () => isolateSelectionSet(set),
      },
    ]),
    {
      id: "exit-isolation",
      label: "Exit selection isolation",
      group: "View",
      aliases: ["restore model", "show context"],
      shortcut: "I",
      enabled: isolatedEntities.length > 0,
      disabledReason: "No active isolation",
      run: exitIsolation,
    },
    { id: "show-all", label: "Show all model elements", group: "View", aliases: ["unhide", "reveal"], run: showAllElements },
    { id: "unlock-all", label: "Unlock all model elements", group: "Modify", aliases: ["release locks"], run: unlockAllElements },
    {
      id: "export-patch",
      label: "Export correction patch",
      group: "Export",
      aliases: ["save changes", "download patch"],
      shortcut: "Ctrl S",
      enabled: Boolean(session.present),
      disabledReason: "Open a PlanGraph first",
      run: () => void exportPatch(),
    },
    {
      id: "discard-changes",
      label: "Discard all session changes",
      group: "File",
      aliases: ["reset", "restore source"],
      enabled: session.past.length > 0 || operations.length > 0,
      disabledReason: "The session has no local changes",
      run: () => setDiscardConfirmOpen(true),
    },
  ];

  const executeStudioCommand = (command: StudioCommand) => {
    if (command.enabled === false) {
      setNotice(command.disabledReason ?? "This command is unavailable for the current selection");
      return;
    }
    command.run();
    setRecentCommandIds((current) => {
      const next = recordRecentCommand(current, command.id);
      localStorage.setItem(RECENT_COMMANDS_KEY, JSON.stringify(next));
      return next;
    });
    setCommandOpen(false);
  };

  if (!graph) {
    return <main className="studio-loading"><DajoongLogo /><h1>Dajoong Studio</h1><p>{notice}</p></main>;
  }

  return (
    <main className="studio-shell">
      <header className="app-header">
        <a className="brand-lockup" href="/"><DajoongLogo compact /><div><strong>DAJOONG</strong><small>Plan2BIM Studio</small></div></a>
        <div className="project-crumb"><span>{graph.project_id ?? "Untitled project"}</span><b>/</b><strong>{graph.sheet_id ?? "PlanGraph"}</strong></div>
        <div className="header-actions">
          <button className="header-button" onClick={() => setConversionOpen(true)}><UploadCloud size={15} /> Convert</button>
          <button className="header-button" onClick={() => fileInput.current?.click()}><FileInput size={15} /> Open files</button>
          {jobId ? <button className="header-button" onClick={() => void downloadJobArtifact(jobId, "ifc")}><Download size={15} /> IFC</button> : null}
          {jobId ? <button className="header-button" onClick={() => void downloadJobArtifact(jobId, "glb")}><Download size={15} /> GLB</button> : null}
          <button className="header-button" onClick={() => downloadJson("corrected-plan-graph.json", graph)}><Download size={15} /> Graph</button>
          <button className="header-button primary" onClick={() => void exportPatch()}><Save size={15} /> Export patch</button>
          {authConfigured ? <button className="header-button icon-only" onClick={() => setAccountOpen(true)} title="Account and privacy"><UserRound size={15} /></button> : null}
          {authConfigured ? <button className="header-button icon-only" onClick={() => void signOut()} title="Sign out"><LogOut size={15} /></button> : null}
          <input ref={fileInput} hidden type="file" multiple accept=".json,image/*" onChange={(event) => void importFiles(event.target.files)} />
        </div>
      </header>
      <nav className="command-bar" aria-label="Editing commands">
        <div className="command-group">
          <button className={activeTool === "select" ? "command active" : "command"} onClick={() => setActiveTool("select")}><BoxSelect size={17} /><span>Select</span></button>
          <button disabled={lockedCollections.includes("walls")} className={activeTool === "wall" ? "command active" : "command"} onClick={() => setActiveTool("wall")}><BrickWall size={17} /><span>Add wall</span></button>
          <button disabled={lockedCollections.includes("openings")} className="command" onClick={addOpening}><DoorOpen size={17} /><span>Add opening</span></button>
          <button disabled={lockedCollections.includes("fixtures")} className={activeTool === "object" ? "command active" : "command"} onClick={() => browseFamilies("insert")}><Box size={17} /><span>Add object</span></button>
          <button disabled={lockedCollections.includes("dimensions")} className={activeTool === "measure" ? "command active" : "command"} onClick={() => setActiveTool("measure")}><Ruler size={17} /><span>Measure</span></button>
        </div>
        <div className="command-divider" />
        <button className="command-search" onClick={() => setCommandOpen(true)}><Search size={14} /><span>Find command</span><kbd>Ctrl K</kbd></button>
        <div className="command-group compact">
          <button className="icon-command" disabled={!session.past.length} onClick={() => dispatch({ type: "undo" })} title="Undo (Ctrl+Z)"><Undo2 size={17} /></button>
          <button className="icon-command" disabled={!session.future.length} onClick={() => dispatch({ type: "redo" })} title="Redo (Ctrl+Y)"><Redo2 size={17} /></button>
          <button className="icon-command" disabled={!canCopyToClipboard} onClick={copySelectionToClipboard} title="Copy BIM selection (Ctrl+C)"><ClipboardCopy size={16} /></button>
          <button className="icon-command" disabled={!canPasteFromClipboard} onClick={pasteBimClipboard} title="Paste BIM selection (Ctrl+V)"><ClipboardPaste size={16} /></button>
        </div>
        <label className="snap-control" title="Plan editing snap increment">
          <Magnet size={14} />
          <span>SNAP</span>
          <select
            value={snapIncrementM}
            onChange={(event) => setSnapIncrementM(Number(event.target.value))}
          >
            <option value={0}>Off</option>
            <option value={0.01}>10 mm</option>
            <option value={0.025}>25 mm</option>
            <option value={0.05}>50 mm</option>
            <option value={0.1}>100 mm</option>
            <option value={0.5}>500 mm</option>
          </select>
        </label>
        <SelectionFilterControl
          exclusions={selectionExclusions}
          counts={selectionFilterCounts}
          onToggle={toggleSelectionFilter}
          onReset={() => {
            setSelectionExclusions([]);
            setNotice("All BIM categories restored to 2D and 3D selection");
          }}
        />
        <button
          className={graph.qualification?.production_release_eligible ? "quality-pill eligible" : "quality-pill"}
          onClick={() => setQualityOpen(true)}
          title="Open model assurance and benchmark evidence"
        >
          <span>{graph.drawing_profile?.difficulty_class ?? "unprofiled"}</span>
          <b>{graph.qualification?.production_release_eligible ? "QUALIFIED" : "REVIEW GATE"}</b>
        </button>
        <div className="level-select"><span>LEVEL</span><select value={levelId} onChange={(event) => { setLevelId(event.target.value); setSelection(null); }}>{graph.levels.map((level) => <option key={level.id} value={level.id}>{level.name}</option>)}</select><button onClick={addLevel} title="Add building level"><Plus size={14} /></button><ChevronDown size={14} /></div>
        <div className="view-switcher">
          <button className={viewMode === "plan" ? "active" : ""} onClick={() => setViewMode("plan")} title="Plan only"><PanelTop size={16} /></button>
          <button className={viewMode === "split" ? "active" : ""} onClick={() => setViewMode("split")} title="Split view"><Columns2 size={16} /></button>
          <button className={viewMode === "model" ? "active" : ""} onClick={() => setViewMode("model")} title="3D only"><Rows2 size={16} /></button>
        </div>
        <button className="review-next" onClick={reviewNext}><Check size={16} /><span>Review next</span><b>{queue.length}</b></button>
      </nav>
      <div className="workspace">
        <ModelTree
          graph={graph}
          levelId={levelId}
          selections={selections}
          reviewOnly={reviewOnly}
          reviewPriorities={reviewPriorities}
          onReviewOnly={setReviewOnly}
          onSelect={onSelect}
          onSelectMany={onSelectTreeRange}
          onOpenContextMenu={openContextMenu}
          hiddenCollections={hiddenCollections}
          lockedCollections={lockedCollections}
          hiddenEntities={hiddenEntities}
          lockedEntities={lockedEntities}
          isolatedEntities={isolatedEntities}
          selectionSets={selectionSets}
          onToggleVisibility={toggleCollectionVisibility}
          onToggleLock={toggleCollectionLock}
          onToggleEntityVisibility={toggleEntityVisibility}
          onToggleEntityLock={toggleEntityLock}
          onExitIsolation={exitIsolation}
          onShowAll={showAllElements}
          onUnlockAll={unlockAllElements}
          onCreateSelectionSet={saveCurrentSelectionSet}
          onRecallSelectionSet={recallSelectionSet}
          onIsolateSelectionSet={isolateSelectionSet}
          onRenameSelectionSet={renameSavedSelectionSet}
          onDeleteSelectionSet={deleteSavedSelectionSet}
        />
        <section className={`canvas-stack ${viewMode}`}>
          {viewMode !== "model" ? (
            <PlanViewport
              graph={graph}
              levelId={levelId}
              selections={selections}
              sourceUrl={sourceUrl}
              snapIncrementM={snapIncrementM}
              activeTool={activeTool}
              hiddenCollections={hiddenCollections}
              lockedCollections={lockedCollections}
              hiddenEntities={hiddenEntities}
              lockedEntities={lockedEntities}
              isolatedEntities={isolatedEntities}
              selectionExclusions={selectionExclusions}
              onSelect={onSelect}
              onSelectMany={onSelectMany}
              onOpenContextMenu={openContextMenu}
              onClearSelection={() => setSelection(null)}
              onBeginGesture={beginGesture}
              onPreviewGesture={previewGesture}
              onCommitGesture={() => dispatch({ type: "commitGesture" })}
              onCancelGesture={() => dispatch({ type: "cancelGesture" })}
              onCreateWall={createWall}
              onCreateMeasurement={(from, to) => {
                if (lockedCollections.includes("dimensions")) {
                  setNotice("Dimensions are locked · unlock them in the Model Browser");
                  return;
                }
                const id = uniqueId(graph.dimensions ?? [], `${levelId}:dimension:manual`);
                const entity: Measurement = {
                  id,
                  level_id: levelId,
                  type: "aligned",
                  name: "Field dimension",
                  from,
                  to,
                  confidence: 1,
                  uncertainty: 0,
                  review_state: "accepted",
                  model_version: "human-correction",
                };
                revealCollection("dimensions");
                dispatch({ type: "add", collection: "dimensions", entity });
                const nextSelection = { collection: "dimensions" as const, id };
                setIsolatedEntities((current) => current.length ? [...current, nextSelection] : current);
                setSelection(nextSelection);
                setNotice("Dimension created · drag either endpoint or enter an exact length");
              }}
              onClearMeasurements={() => {
                if (lockedCollections.includes("dimensions")) {
                  setNotice("Dimensions are locked · clear blocked");
                  return;
                }
                const dimensions = (graph.dimensions ?? [])
                  .filter((item) => item.level_id === levelId)
                  .map((item) => ({ collection: "dimensions" as const, id: item.id }));
                if (!dimensions.length) return;
                dispatch({ type: "batchDelete", selections: dimensions });
                setSelection(null);
                setNotice(`${dimensions.length} dimension${dimensions.length === 1 ? "" : "s"} cleared`);
              }}
              placementFamily={placementFamily}
              onPlaceFixture={placePendingFixture}
              onCancelFixturePlacement={cancelFixturePlacement}
              onChangeFixtureFamily={() => browseFamilies("insert")}
              rehostOpeningId={rehostOpeningId}
              onPickOpeningHost={pickOpeningHost}
              onCancelOpeningRehost={cancelOpeningRehost}
            />
          ) : null}
          {viewMode !== "plan" ? (
            <ModelViewport
              graph={graph}
              levelId={levelId}
              selections={selections}
              hiddenCollections={hiddenCollections}
              lockedCollections={lockedCollections}
              hiddenEntities={hiddenEntities}
              lockedEntities={lockedEntities}
              isolatedEntities={isolatedEntities}
              selectionExclusions={selectionExclusions}
              snapIncrementM={snapIncrementM}
              onTransformCommit={commitModelTransform}
              onIsolateSelection={isolateSelection}
              onExitIsolation={exitIsolation}
              onSelect={onSelect}
              onOpenContextMenu={openContextMenu}
            />
          ) : null}
          <SelectionActionBar
            count={selections.length}
            canDuplicate={canDuplicate}
            canCopy={canCopyToClipboard}
            canPaste={canPasteFromClipboard}
            canMoveExact={canExactMove}
            canRotateExact={canExactRotate}
            canPattern={canRepeatPattern}
            canAlign={canArrange}
            canDistribute={canDistribute}
            keyObjectLabel={selection?.id ?? "none"}
            canFlipDoor={canFlipDoor}
            canRehostOpening={canRehostOpening}
            canJoinWalls={canJoinWalls}
            copyTargets={graph.levels.filter((level) => level.id !== levelId)}
            locked={selectionLocked}
            onIsolate={isolateSelection}
            onDuplicate={duplicateSelection}
            onCopy={copySelectionToClipboard}
            onPaste={pasteBimClipboard}
            onMoveExact={() => setExactMoveOpen(true)}
            onRotateExact={() => setExactRotateOpen(true)}
            onMirror={() => setPatternOpen("mirror")}
            onArray={() => setPatternOpen("array")}
            onAlign={alignSelection}
            onDistribute={distributeSelection}
            onFlipDoorHanding={flipDoorHanding}
            onRehostOpening={startOpeningRehost}
            onReverseDoorSwing={reverseDoorSwing}
            onJoinWalls={joinSelectedWalls}
            onCornerWalls={cornerSelectedWalls}
            onCopyToLevel={copySelectionToLevel}
            onAccept={acceptSelected}
            onDelete={deleteSelected}
          />
          <div className="status-toast"><span className="status-dot accepted" />{notice}<b>{selections.length > 1 ? `${selections.length} elements selected` : selectionMetric(selectedEntity) || (operations.length ? `${operations.length} changes` : "No changes")}</b></div>
        </section>
        <PropertyPanel
          entity={selectedEntity}
          selection={selection}
          entities={selectedEntities}
          selections={selections}
          changeCount={operations.filter((item) => item.entity_id === selection?.id).length}
          reviewPriority={selectedReviewPriority}
          locked={selectionLocked}
          onEdit={(changes) => selection && editSelection(selection, changes)}
          onBatchEdit={editSelections}
          onAccept={acceptSelected}
          onBatchAccept={acceptSelected}
          onDelete={() => {
            if (!selection) return;
            if (
              selection.collection === "walls" &&
              (graph.constraints ?? []).some((constraint) =>
                constraint.references.some((reference) => reference.entity_id === selection.id),
              )
            ) {
              setNotice("This wall has geometric constraints · delete the constraints first");
              return;
            }
            if (selection.collection === "walls" && graph.openings.some((item) => item.wall_id === selection.id)) {
              setNotice("This wall hosts openings · rehost or delete them first");
              return;
            }
            if (selection.collection === "rooms" && graph.fixtures.some((item) => item.room_id === selection.id)) {
              setNotice("This room contains objects · reassign them first");
              return;
            }
            dispatch({ type: "delete", selection });
            setSelection(null);
          }}
          onBatchDelete={() => {
            const selectedIds = new Set(selections.map((item) => item.id));
            const blockedWall = selections.some(
              (item) =>
                item.collection === "walls" &&
                graph.openings.some(
                  (opening) => opening.wall_id === item.id && !selectedIds.has(opening.id),
                ),
            );
            const blockedRoom = selections.some(
              (item) =>
                item.collection === "rooms" &&
                graph.fixtures.some(
                  (fixture) => fixture.room_id === item.id && !selectedIds.has(fixture.id),
                ),
            );
            const blockedConstraint = selections.some(
              (item) =>
                item.collection === "walls" &&
                (graph.constraints ?? []).some(
                  (constraint) =>
                    constraint.references.some((reference) => reference.entity_id === item.id) &&
                    !selectedIds.has(constraint.id),
                ),
            );
            if (blockedWall || blockedRoom || blockedConstraint) {
              setNotice("Selection contains hosted elements · include or rehost them first");
              return;
            }
            dispatch({ type: "batchDelete", selections });
            setSelection(null);
          }}
          onBrowseFamily={() => browseFamilies("replace")}
          onRequestOpeningRehost={startOpeningRehost}
        />
      </div>
      <HistoryTimeline
        entries={historyEntries}
        collapsed={historyCollapsed}
        onToggleCollapsed={toggleHistoryCollapsed}
        onJump={jumpToHistory}
      />
      {contextMenu ? (
        <ElementContextMenu
          request={contextMenu}
          title={String(
            contextEntity?.name ??
            contextEntity?.type ??
            contextEntity?.family_id ??
            contextMenu.anchor.collection.replaceAll("_", " "),
          )}
          hidden={contextHidden}
          locked={contextLocked}
          allLocked={contextAllIndividuallyLocked}
          inheritedLock={contextInheritedLock}
          accepted={contextAccepted}
          canDuplicate={contextCanDuplicate}
          canCopy={canCopyToClipboard}
          canPaste={canPasteFromClipboard}
          canRehostOpening={contextCanRehostOpening}
          canArrange={canArrange}
          canDistribute={canDistribute}
          keyObjectLabel={selection?.id ?? contextMenu.anchor.id}
          relatedActions={contextRelatedActions}
          onClose={() => setContextMenu(null)}
          onIsolate={() => isolateItems(contextMenu.targets)}
          onToggleVisibility={toggleContextVisibility}
          onToggleLock={toggleContextLock}
          onDuplicate={duplicateSelection}
          onCopy={copySelectionToClipboard}
          onPaste={pasteBimClipboard}
          onRehostOpening={startOpeningRehost}
          onAlign={alignSelection}
          onDistribute={distributeSelection}
          onAccept={acceptSelected}
          onDelete={deleteSelected}
          onSelectRelated={selectRelatedElements}
        />
      ) : null}
      {conversionOpen ? (
        <Suspense fallback={<StudioToolFallback />}>
          <ConversionDialog
            open
            onClose={() => setConversionOpen(false)}
            onStatus={setNotice}
            onComplete={(convertedGraph, convertedSourceUrl, convertedJobId) => {
              rehostViewStateRef.current = null;
              setRehostOpeningId(null);
              dispatch({ type: "load", graph: convertedGraph });
              setHiddenCollections([]);
              setLockedCollections([]);
              setHiddenEntities([]);
              setLockedEntities([]);
              setIsolatedEntities([]);
              setSelectionSets([]);
              setLevelId(convertedGraph.levels[0]?.id ?? "L1");
              setSelection(null);
              setJobId(convertedJobId);
              if (convertedSourceUrl) {
                if (sourceUrl.startsWith("blob:")) URL.revokeObjectURL(sourceUrl);
                setSourceUrl(convertedSourceUrl);
              }
            }}
          />
        </Suspense>
      ) : null}
      {accountOpen ? <AccountDialog onClose={() => setAccountOpen(false)} /> : null}
      {exactMoveOpen ? (
        <Suspense fallback={<StudioToolFallback />}>
          <ExactMoveDialog open count={selections.length} onClose={() => setExactMoveOpen(false)} onApply={applyExactTranslation} />
        </Suspense>
      ) : null}
      {exactRotateOpen ? (
        <Suspense fallback={<StudioToolFallback />}>
          <ExactRotateDialog
            open
            count={selections.length}
            defaultPivot={exactRotationPivot}
            onClose={() => setExactRotateOpen(false)}
            onApply={applyExactRotation}
          />
        </Suspense>
      ) : null}
      {patternOpen ? (
        <Suspense fallback={<StudioToolFallback />}>
          <PatternDialog
            open={patternOpen}
            selectionCount={selections.length}
            defaultCenter={repeatPatternCenter}
            onClose={() => setPatternOpen(null)}
            onMirror={applyMirrorPattern}
            onArray={applyLinearArray}
          />
        </Suspense>
      ) : null}
      {commandOpen ? (
        <Suspense fallback={<StudioToolFallback />}>
          <CommandPalette
            open
            commands={studioCommands}
            recentCommandIds={recentCommandIds}
            onClose={() => setCommandOpen(false)}
            onExecute={executeStudioCommand}
          />
        </Suspense>
      ) : null}
      {familyBrowserOpen ? (
        <Suspense fallback={<StudioToolFallback />}>
          <FamilyBrowser
            open
            mode={familyBrowserMode}
            selectionCount={selections.filter((item) => item.collection === "fixtures").length}
            error={familyBrowserError || undefined}
            onClose={() => { setFamilyBrowserOpen(false); setFamilyBrowserError(""); }}
            onApply={placeFamily}
          />
        </Suspense>
      ) : null}
      {qualityOpen ? (
        <Suspense fallback={<StudioToolFallback />}>
          <QualityReview
            graph={graph}
            open
            reviewCount={queue.length}
            reviewPriorities={reviewPriorities}
            onClose={() => setQualityOpen(false)}
            onReviewNext={reviewNext}
            onLocateEntities={locateFindingEntities}
          />
        </Suspense>
      ) : null}
      {discardConfirmOpen ? (
        <div
          className="dialog-backdrop"
          role="presentation"
          onMouseDown={(event) => event.target === event.currentTarget && setDiscardConfirmOpen(false)}
        >
          <section className="conversion-dialog discard-dialog" role="dialog" aria-label="Discard session changes">
            <div className="dialog-header">
              <div><span className="eyebrow">LOCAL SESSION</span><h2>Restore the imported model?</h2></div>
            </div>
            <div className="discard-dialog-copy">
              <p>This removes every unexported correction, constraint, and dimension from this browser session.</p>
              <strong>The imported source and generated artifacts are not deleted.</strong>
            </div>
            <div className="dialog-actions">
              <button className="secondary-button" onClick={() => setDiscardConfirmOpen(false)}>Cancel</button>
              <button
                className="primary-button destructive"
                onClick={() => {
                  dispatch({ type: "resetToSource" });
                  setSelection(null);
                  setDiscardConfirmOpen(false);
                  setNotice("All local changes discarded · imported model restored");
                }}
              >
                Discard changes
              </button>
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}

const openingGeometryFields = new Set(["center_m", "width_m", "wall_id", "x_m"]);
const wallGeometryFields = new Set(["from", "to"]);
const fixturePlacementFields = new Set([
  "center_m",
  "size_m",
  "yaw_deg",
  "family_id",
  "mounting",
  "host_wall_id",
]);

function hasOpeningGeometryChange(changes: Record<string, unknown>): boolean {
  return Object.keys(changes).some((key) => openingGeometryFields.has(key));
}

function hasWallGeometryChange(changes: Record<string, unknown>): boolean {
  return Object.keys(changes).some((key) => wallGeometryFields.has(key));
}

export interface PreparedBimTransform {
  valid: boolean;
  selections: Selection[];
  changesById: EntityChanges;
  notices: string[];
  reason?: string;
}

/**
 * Resolves a requested transform against BIM relationships before it reaches
 * the reducer. Wall edits are expanded to constrained endpoints, hosted
 * openings, room boundaries, and wall-mounted fixtures; ordinary component
 * transforms still use the shared placement and clearance solver.
 */
export function prepareBimTransform(
  graph: PlanGraph,
  items: Selection[],
  changesById: EntityChanges,
): PreparedBimTransform {
  const wallChangesById: EntityChanges = {};
  for (const item of items) {
    const changes = changesById[item.id];
    if (item.collection === "walls" && changes && hasWallGeometryChange(changes)) {
      wallChangesById[item.id] = changes;
    }
  }

  if (!Object.keys(wallChangesById).length) {
    const validation = validateFixtureTransformChanges(graph, changesById);
    return validation.valid
      ? {
          valid: true,
          selections: items,
          changesById: validation.changesById,
          notices: validation.notices,
        }
      : {
          valid: false,
          selections: items,
          changesById: {},
          notices: validation.notices,
          reason: validation.reason,
        };
  }

  const plan = planWallTransform(graph, wallChangesById, changesById);
  if (!plan.valid) {
    return {
      valid: false,
      selections: items,
      changesById: {},
      notices: plan.notices,
      reason: plan.reason,
    };
  }

  const entries = new Map<string, WallTransformEntry>();
  for (const item of items) {
    const changes = changesById[item.id];
    if (!changes) continue;
    entries.set(selectionKey(item), { selection: item, changes });
  }
  for (const entry of plan.entries) {
    const key = selectionKey(entry.selection);
    const current = entries.get(key);
    entries.set(key, {
      selection: entry.selection,
      changes: { ...(current?.changes ?? {}), ...entry.changes },
    });
  }
  const resolved = [...entries.values()];
  return {
    valid: true,
    selections: resolved.map((entry) => entry.selection),
    changesById: Object.fromEntries(resolved.map((entry) => [entry.selection.id, entry.changes])),
    notices: plan.notices,
  };
}

function hasFixturePlacementChange(changes: Record<string, unknown>): boolean {
  return Object.keys(changes).some((key) => fixturePlacementFields.has(key));
}

function openingPlacementNotice(reason?: string, conflictId?: string): string {
  if (reason === "overlap") {
    return `Opening edit blocked · overlaps ${conflictId ?? "another hosted opening"}`;
  }
  if (reason === "outside_wall") return "Opening edit blocked · it must remain inside the host wall";
  if (reason === "too_narrow") return "Opening edit blocked · minimum width is 200 mm";
  return "Opening edit blocked · host wall geometry is invalid";
}

function correctionChangesForAddedEntity(entity: Record<string, unknown>): Record<string, unknown> {
  const ignored = new Set([
    "id",
    "confidence",
    "uncertainty",
    "review_state",
    "model_version",
    "correction_id",
    "reviewed_by",
  ]);
  return Object.fromEntries(Object.entries(entity).filter(([key]) => !ignored.has(key)));
}

function editorToolCollection(tool: EditorTool): CollectionName | null {
  if (tool === "wall") return "walls";
  if (tool === "opening") return "openings";
  if (tool === "room") return "rooms";
  if (tool === "object") return "fixtures";
  if (tool === "measure") return "dimensions";
  return null;
}

function actionCollections(action: SessionAction): CollectionName[] {
  if (action.type === "add") return [action.collection];
  if (action.type === "addMany") return [...new Set(action.items.map((item) => item.collection))];
  if (action.type === "constrainWalls") {
    return [...new Set(["constraints" as const, ...action.selections.map((item) => item.collection)])];
  }
  if (action.type === "previewTransform") {
    return [...new Set(action.entries.map((entry) => entry.selection.collection))];
  }
  if (
    action.type === "batchEdit" ||
    action.type === "batchTransform" ||
    action.type === "batchAccept" ||
    action.type === "batchDelete"
  ) {
    return [...new Set(action.selections.map((item) => item.collection))];
  }
  if (
    action.type === "edit" ||
    action.type === "accept" ||
    action.type === "delete" ||
    action.type === "beginGesture" ||
    action.type === "previewGesture"
  ) {
    return [action.selection.collection];
  }
  return [];
}

export function actionSelections(action: SessionAction): Selection[] {
  if (action.type === "previewTransform") return action.entries.map((entry) => entry.selection);
  if (
    action.type === "batchEdit" ||
    action.type === "batchTransform" ||
    action.type === "batchAccept" ||
    action.type === "batchDelete" ||
    action.type === "constrainWalls"
  ) {
    return action.selections;
  }
  if (
    action.type === "edit" ||
    action.type === "accept" ||
    action.type === "delete" ||
    action.type === "beginGesture" ||
    action.type === "previewGesture"
  ) {
    return [action.selection];
  }
  return [];
}

export function actionMutationSelections(
  action: SessionAction,
  graph: PlanGraph | null,
): Selection[] {
  const direct = actionSelections(action);
  if (!graph) return direct;

  const endpointSeeds: Array<{ id: string; handle: "from" | "to" }> = [];
  const collectWallEndpoints = (
    item: Selection,
    changes: Record<string, unknown> | undefined,
  ) => {
    if (item.collection !== "walls" || !changes) return;
    if (Object.prototype.hasOwnProperty.call(changes, "from")) {
      endpointSeeds.push({ id: item.id, handle: "from" });
    }
    if (Object.prototype.hasOwnProperty.call(changes, "to")) {
      endpointSeeds.push({ id: item.id, handle: "to" });
    }
  };

  if (action.type === "edit" || action.type === "previewGesture") {
    collectWallEndpoints(action.selection, action.changes);
  } else if (action.type === "batchEdit") {
    action.selections.forEach((item) => collectWallEndpoints(item, action.changes));
  } else if (action.type === "batchTransform" || action.type === "constrainWalls") {
    action.selections.forEach((item) =>
      collectWallEndpoints(item, action.changesById[item.id]),
    );
  }

  if (!endpointSeeds.length) return direct;
  const queue = [...endpointSeeds];
  const visitedHandles = new Set<string>();
  const affectedWallIds = new Set<string>();
  while (queue.length) {
    const current = queue.shift();
    if (!current) break;
    const key = `${current.id}:${current.handle}`;
    if (visitedHandles.has(key)) continue;
    visitedHandles.add(key);
    affectedWallIds.add(current.id);
    for (const constraint of graph.constraints ?? []) {
      if (
        constraint.type !== "coincident" ||
        !constraint.references.some(
          (reference) =>
            reference.entity_id === current.id && reference.handle === current.handle,
        )
      ) {
        continue;
      }
      for (const reference of constraint.references) {
        queue.push({ id: reference.entity_id, handle: reference.handle });
      }
    }
  }

  const expanded: Selection[] = [
    ...direct,
    ...[...affectedWallIds].map((id) => ({ collection: "walls" as const, id })),
    ...graph.openings
      .filter((opening) => affectedWallIds.has(opening.wall_id))
      .map((opening) => ({ collection: "openings" as const, id: opening.id })),
  ];
  const seen = new Set<string>();
  return expanded.filter((item) => {
    const key = selectionKey(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function selectionMetric(entity: ReturnType<typeof findEntity>): string {
  if (!entity) return "";
  const from = entity.from;
  const to = entity.to;
  if (Array.isArray(from) && Array.isArray(to) && from.length >= 2 && to.length >= 2) {
    return `${Math.hypot(Number(to[0]) - Number(from[0]), Number(to[1]) - Number(from[1])).toFixed(2)} m`;
  }
  const polygon = entity.polygon;
  if (Array.isArray(polygon) && polygon.length > 2 && polygon.every((point) => Array.isArray(point) && point.length >= 2)) {
    const points = polygon as [number, number][];
    const area = Math.abs(points.reduce((sum, point, index) => {
      const next = points[(index + 1) % points.length];
      return sum + point[0] * next[1] - next[0] * point[1];
    }, 0)) / 2;
    return `${area.toFixed(1)} m²`;
  }
  const size = entity.size_m;
  if (Array.isArray(size)) return size.map((value) => Number(value).toFixed(2)).join(" × ") + " m";
  if (typeof entity.width_m === "number" && typeof entity.height_m === "number") return `${entity.width_m.toFixed(2)} × ${entity.height_m.toFixed(2)} m`;
  return "";
}

function defaultFixtureElevation(family: FixtureFamily, level?: LevelEntity): number {
  if (family.mounting === "ceiling") {
    return Math.max(0, (level?.nominal_height_m ?? 3) - family.size_m[2]);
  }
  if (family.type === "receptacle") return 0.3;
  if (family.type === "electrical-panel") return 0.4;
  if (family.type === "sink") return 0.75;
  if (family.type === "fan-coil") return 2.1;
  return 0;
}

function uniqueId(items: Array<{ id: string }>, base: string): string {
  let index = 1;
  while (items.some((item) => item.id === `${base}:${index}`)) index += 1;
  return `${base}:${index}`;
}

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, canonical(item)]));
  }
  return value;
}

async function contentHash(graph: PlanGraph): Promise<string> {
  const payload = structuredClone(graph);
  if (payload.pipeline) delete payload.pipeline.content_sha256;
  const bytes = new TextEncoder().encode(JSON.stringify(canonical(payload)));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest)).map((item) => item.toString(16).padStart(2, "0")).join("");
}

async function downloadJobArtifact(jobId: string, artifact: "ifc" | "glb") {
  const response = await authFetch(studioApiUrl(`/api/jobs/${jobId}/artifacts/${artifact}`));
  if (!response.ok) throw new Error(`Could not download ${artifact.toUpperCase()}`);
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `dajoong-model.${artifact}`;
  anchor.click();
  URL.revokeObjectURL(url);
}
