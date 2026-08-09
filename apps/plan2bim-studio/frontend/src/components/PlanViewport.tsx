import { Box, Focus, Maximize2, Minus, Plus, RotateCw, Search, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  PointerEvent as ReactPointerEvent,
  WheelEvent as ReactWheelEvent,
} from "react";

import {
  distanceMeters,
  movePolygonVertex,
  smartSnap,
  type SmartGuide,
} from "../editorGeometry";
import { selectionKey } from "../editorViewState";
import type { FixtureFamily } from "../families";
import {
  fixturePlacementAt,
  fixturePlacementMessage,
  validateFixtureEntityChanges,
  type FixturePlacement,
} from "../fixturePlacement";
import { graphBounds } from "../graph";
import {
  moveOpeningToPoint,
  openingFrame,
  resizeOpeningFromEdge,
  validateOpeningPlacement,
} from "../openingGeometry";
import {
  insertRoomBoundaryVertex,
  removeRoomBoundaryVertex,
  snapRoomBoundaryPoint,
} from "../roomBoundary";
import {
  endpointFromLengthAngle,
  lengthAndAngle,
  parseAngleInput,
  parseLengthInput,
} from "../precisionInput";
import {
  fittedPlanView,
  panPlanViewByPixels,
  planViewBoxValue,
  planZoomPercent,
  viewForSelections,
  zoomPlanViewAt,
  type PlanViewBox,
} from "../planNavigation";
import {
  cycleSelectionIndex,
  hitTestPlanGraph,
  selectionCandidateLabel,
} from "../planHitTest";
import {
  elementFootprint,
  selectInRectangle,
  selectionMode,
  selectionRectangle,
} from "../windowSelection";
import type {
  CollectionName,
  EditorTool,
  FixtureEntity,
  OpeningEntity,
  PlanGraph,
  RoomEntity,
  Selection,
  WallEntity,
} from "../types";

interface PlanViewportProps {
  graph: PlanGraph;
  levelId: string;
  selections: Selection[];
  sourceUrl?: string;
  snapIncrementM: number;
  activeTool: EditorTool;
  hiddenCollections: CollectionName[];
  lockedCollections: CollectionName[];
  hiddenEntities: Selection[];
  lockedEntities: Selection[];
  isolatedEntities: Selection[];
  selectionExclusions: CollectionName[];
  onSelect: (selection: Selection, additive?: boolean) => void;
  onSelectMany: (selections: Selection[], additive?: boolean) => void;
  onOpenContextMenu: (selection: Selection, clientX: number, clientY: number) => void;
  onClearSelection: () => void;
  onBeginGesture: (selection: Selection) => boolean;
  onPreviewGesture: (selection: Selection, changes: Record<string, unknown>) => boolean;
  onCommitGesture: () => void;
  onCancelGesture: () => void;
  onCreateWall: (from: [number, number], to: [number, number]) => void;
  onCreateMeasurement: (from: [number, number], to: [number, number]) => void;
  onClearMeasurements: () => void;
  placementFamily: FixtureFamily | null;
  onPlaceFixture: (placement: FixturePlacement) => void;
  onCancelFixturePlacement: () => void;
  onChangeFixtureFamily: () => void;
  rehostOpeningId: string | null;
  onPickOpeningHost: (wallId: string, point: [number, number]) => void;
  onCancelOpeningRehost: () => void;
}

interface DragState {
  selection: Selection;
  mode: "fixture" | "fixture-rotate" | "opening" | "opening-start" | "opening-end" | "vertical" | "wall-from" | "wall-to" | "room-vertex" | "measurement-from" | "measurement-to";
  origin?: [number, number];
  anchor?: [number, number];
  vertexIndex?: number;
}

interface SelectionWindowState {
  start: [number, number];
  current: [number, number];
  additive: boolean;
  pointerId: number;
  clickSelection?: Selection;
}

interface NavigationDragState {
  pointerId: number;
  lastClientX: number;
  lastClientY: number;
}

interface PinchState {
  pointerIds: [number, number];
  startDistance: number;
  startCenter: [number, number];
  anchorWorld: [number, number];
  startView: PlanViewBox;
}

interface HoverCycleState {
  point: [number, number];
  cursor: [number, number];
  hud: [number, number];
  candidates: Selection[];
  index: number;
}

interface LastCyclePick {
  signature: string;
  clientX: number;
  clientY: number;
  timestamp: number;
  index: number;
}

export function PlanViewport({
  graph,
  levelId,
  selections,
  sourceUrl,
  snapIncrementM,
  activeTool,
  hiddenCollections,
  lockedCollections,
  hiddenEntities,
  lockedEntities,
  isolatedEntities,
  selectionExclusions,
  onSelect,
  onSelectMany,
  onOpenContextMenu,
  onClearSelection,
  onBeginGesture,
  onPreviewGesture,
  onCommitGesture,
  onCancelGesture,
  onCreateWall,
  onCreateMeasurement,
  onClearMeasurements,
  placementFamily,
  onPlaceFixture,
  onCancelFixturePlacement,
  onChangeFixtureFamily,
  rehostOpeningId,
  onPickOpeningHost,
  onCancelOpeningRehost,
}: PlanViewportProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const lengthInputRef = useRef<HTMLInputElement>(null);
  const angleInputRef = useRef<HTMLInputElement>(null);
  const navigationDragRef = useRef<NavigationDragState | null>(null);
  const touchPointersRef = useRef(new Map<number, [number, number]>());
  const pinchRef = useRef<PinchState | null>(null);
  const lastCyclePickRef = useRef<LastCyclePick | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [selectionWindow, setSelectionWindow] = useState<SelectionWindowState | null>(null);
  const [spacePan, setSpacePan] = useState(false);
  const [isPanning, setIsPanning] = useState(false);
  const [hoverCycle, setHoverCycle] = useState<HoverCycleState | null>(null);
  const [constructionStart, setConstructionStart] = useState<[number, number] | null>(null);
  const [cursorPoint, setCursorPoint] = useState<[number, number] | null>(null);
  const [placementCursor, setPlacementCursor] = useState<[number, number] | null>(null);
  const [placementYaw, setPlacementYaw] = useState(0);
  const [guides, setGuides] = useState<SmartGuide[]>([]);
  const [snapLabel, setSnapLabel] = useState("");
  const [precisionLength, setPrecisionLength] = useState("");
  const [precisionAngle, setPrecisionAngle] = useState("");
  const bounds = useMemo(() => graphBounds(graph, levelId), [graph, levelId]);
  const selection = selections.at(-1) ?? null;
  const isSelected = (collection: Selection["collection"], id: string) =>
    selections.some((item) => item.collection === collection && item.id === id);
  const hidden = useMemo(() => new Set(hiddenCollections), [hiddenCollections]);
  const locked = useMemo(() => new Set(lockedCollections), [lockedCollections]);
  const hiddenEntityKeys = useMemo(
    () => new Set(hiddenEntities.map(selectionKey)),
    [hiddenEntities],
  );
  const lockedEntityKeys = useMemo(
    () => new Set(lockedEntities.map(selectionKey)),
    [lockedEntities],
  );
  const isolatedEntityKeys = useMemo(
    () => new Set(isolatedEntities.map(selectionKey)),
    [isolatedEntities],
  );
  const selectionExclusionSet = useMemo(
    () => new Set(selectionExclusions),
    [selectionExclusions],
  );
  const isEntityHidden = (collection: CollectionName, id: string) =>
    hiddenEntityKeys.has(selectionKey({ collection, id })) ||
    (isolatedEntityKeys.size > 0 && !isolatedEntityKeys.has(selectionKey({ collection, id })));
  const isEntityLocked = (collection: CollectionName, id: string) =>
    locked.has(collection) || lockedEntityKeys.has(selectionKey({ collection, id }));
  const width = Math.max(1, bounds.maxX - bounds.minX);
  const height = Math.max(1, bounds.maxY - bounds.minY);
  const padding = Math.max(width, height) * 0.06;
  const fittedView = useMemo(() => fittedPlanView(bounds), [bounds]);
  const [planView, setPlanView] = useState<PlanViewBox>(fittedView);
  const viewBox = planViewBoxValue(planView);
  const zoomPercent = planZoomPercent(fittedView, planView);
  const documentViewKey = `${graph.project_id ?? "project"}:${graph.sheet_id ?? "sheet"}:${levelId}`;
  const fixturePlacement = useMemo(
    () => placementFamily && placementCursor
      ? fixturePlacementAt(graph, placementFamily, levelId, placementCursor, placementYaw)
      : null,
    [graph, levelId, placementCursor, placementFamily, placementYaw],
  );
  const rotatePlacement = useCallback((reverse = false) => {
    const step = placementFamily?.mounting === "wall" ? 180 : reverse ? 270 : 90;
    setPlacementYaw((current) => (current + step) % 360);
  }, [placementFamily?.mounting]);

  useEffect(() => {
    if (activeTool === "object" && fixturePlacement) {
      setSnapLabel(fixturePlacementMessage(fixturePlacement));
    }
  }, [activeTool, fixturePlacement]);

  const fitAll = useCallback(() => {
    setPlanView(fittedView);
    setSnapLabel("Fit All · entire level visible");
  }, [fittedView]);

  const fitSelection = useCallback(() => {
    if (!selections.length) {
      fitAll();
      return;
    }
    setPlanView(viewForSelections(graph, selections, fittedView));
    setSnapLabel(`Fit Selection · ${selections.length} element${selections.length === 1 ? "" : "s"}`);
  }, [fitAll, fittedView, graph, selections]);

  useEffect(() => {
    setConstructionStart(null);
    setCursorPoint(null);
    setGuides([]);
    setPrecisionLength("");
    setPrecisionAngle("");
    setSelectionWindow(null);
    setHoverCycle(null);
    lastCyclePickRef.current = null;
  }, [activeTool, levelId]);

  useEffect(() => {
    setPlacementCursor(null);
    setPlacementYaw(0);
  }, [levelId, placementFamily?.id]);

  useEffect(() => {
    setHoverCycle(null);
    lastCyclePickRef.current = null;
  }, [selectionExclusions]);

  useEffect(() => {
    setPlanView(fittedView);
    navigationDragRef.current = null;
    touchPointersRef.current.clear();
    pinchRef.current = null;
    setIsPanning(false);
    setHoverCycle(null);
    lastCyclePickRef.current = null;
  // Reset only when the drawing or level changes, not after each geometry edit.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentViewKey]);

  const toWorld = (
    event: ReactPointerEvent<SVGSVGElement>,
    origin?: [number, number],
  ) => {
    const svg = svgRef.current;
    if (!svg) return { point: [0, 0] as [number, number], guides: [], label: "" };
    const transformed = toRawWorld(event);
    if (!transformed) return { point: [0, 0] as [number, number], guides: [], label: "" };
    const rect = svg.getBoundingClientRect();
    const worldPerPixel = Math.max(
      planView.width / Math.max(1, rect.width),
      planView.height / Math.max(1, rect.height),
    );
    return smartSnap(
      transformed,
      graph,
      levelId,
      {
        grid_m: snapIncrementM,
        tolerance_m: worldPerPixel * 12,
        origin,
        orthogonal: event.shiftKey,
        disabled: event.altKey,
      },
    );
  };

  const toRawWorld = (
    event: { clientX: number; clientY: number },
  ): [number, number] | null => {
    const svg = svgRef.current;
    if (!svg) return null;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const transformed = point.matrixTransform(svg.getScreenCTM()?.inverse());
    return [transformed.x, transformed.y];
  };

  const zoomAt = (anchor: [number, number], factor: number) => {
    setPlanView((current) => zoomPlanViewAt(
      current,
      anchor,
      factor,
      fittedView.width / 128,
      fittedView.width * 8,
    ));
  };

  const zoomFromCenter = (factor: number) => {
    zoomAt(
      [planView.x + planView.width / 2, planView.y + planView.height / 2],
      factor,
    );
  };

  const onPlanWheel = (event: ReactWheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    const anchor = toRawWorld(event);
    if (!anchor) return;
    zoomAt(anchor, Math.exp(event.deltaY * 0.0015));
  };

  const resolvePrecisionEndpoint = (
    rawPoint: [number, number],
  ): [number, number] | null => {
    if (!constructionStart) return rawPoint;
    const current = lengthAndAngle(constructionStart, rawPoint);
    const enteredLength = precisionLength.trim()
      ? parseLengthInput(precisionLength)
      : current.lengthM;
    const enteredAngle = precisionAngle.trim()
      ? parseAngleInput(precisionAngle)
      : current.angleDeg;
    if (enteredLength === null || enteredAngle === null) return null;
    return endpointFromLengthAngle(constructionStart, enteredLength, enteredAngle);
  };

  const precisionPreview = constructionStart && cursorPoint
    ? resolvePrecisionEndpoint(cursorPoint)
    : null;
  const currentPrecision = constructionStart && cursorPoint
    ? lengthAndAngle(constructionStart, cursorPoint)
    : { lengthM: 0, angleDeg: 0 };
  const precisionValid = Boolean(
    precisionPreview
    && (!precisionLength.trim() || parseLengthInput(precisionLength) !== null)
    && (!precisionAngle.trim() || parseAngleInput(precisionAngle) !== null),
  );

  const resetConstruction = () => {
    setConstructionStart(null);
    setCursorPoint(null);
    setGuides([]);
    setPrecisionLength("");
    setPrecisionAngle("");
    setSnapLabel("");
  };

  const completeConstruction = (point: [number, number] | null) => {
    if (!constructionStart || !point || distanceMeters(constructionStart, point) < 0.05) {
      setSnapLabel("Enter a valid length of at least 50 mm");
      return;
    }
    if (activeTool === "wall") {
      onCreateWall(constructionStart, point);
      setConstructionStart(point);
      setCursorPoint(point);
      setGuides([]);
      setPrecisionLength("");
      setPrecisionAngle("");
      setSnapLabel("Wall placed · continue from endpoint or press Esc");
      return;
    }
    onCreateMeasurement(constructionStart, point);
    resetConstruction();
  };

  useEffect(() => {
    if (!constructionStart || (activeTool !== "wall" && activeTool !== "measure")) return;
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const inPrecisionHud = Boolean(target?.closest(".precision-hud"));
      if (inPrecisionHud) return;
      const consume = () => {
        event.preventDefault();
        event.stopImmediatePropagation();
      };
      if (event.key === "Escape") {
        consume();
        resetConstruction();
        return;
      }
      if (event.key === "Tab") {
        consume();
        lengthInputRef.current?.focus();
        lengthInputRef.current?.select();
        return;
      }
      if (event.key === "Enter") {
        consume();
        completeConstruction(precisionPreview);
        return;
      }
      if (event.key.toLowerCase() === "l") {
        consume();
        lengthInputRef.current?.focus();
        lengthInputRef.current?.select();
        return;
      }
      if (event.key.toLowerCase() === "a") {
        consume();
        angleInputRef.current?.focus();
        angleInputRef.current?.select();
        return;
      }
      if (event.key === "Backspace") {
        consume();
        setPrecisionLength((value) => value.slice(0, -1));
        return;
      }
      if (/^[0-9.]$/.test(event.key)) {
        consume();
        setPrecisionLength((value) => `${value}${event.key}`);
        window.requestAnimationFrame(() => lengthInputRef.current?.focus());
      }
    };
    window.addEventListener("keydown", handler, true);
    return () => window.removeEventListener("keydown", handler, true);
  });

  useEffect(() => {
    const isTyping = (target: EventTarget | null) => {
      const element = target as HTMLElement | null;
      return element?.tagName === "INPUT"
        || element?.tagName === "TEXTAREA"
        || element?.tagName === "SELECT"
        || Boolean(element?.isContentEditable);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTyping(event.target)) return;
      if (event.code === "Space") {
        event.preventDefault();
        setSpacePan(true);
        return;
      }
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key.toLowerCase() === "f") {
        event.preventDefault();
        fitSelection();
        return;
      }
      if (event.key === "Home") {
        event.preventDefault();
        fitAll();
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === "Space") setSpacePan(false);
    };
    const onBlur = () => {
      setSpacePan(false);
      navigationDragRef.current = null;
      setIsPanning(false);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
    };
  }, [fitAll, fitSelection]);

  useEffect(() => {
    if (!selectionWindow) return;
    const cancelWindow = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setSelectionWindow(null);
      setSnapLabel("Selection window cancelled");
    };
    window.addEventListener("keydown", cancelWindow, true);
    return () => window.removeEventListener("keydown", cancelWindow, true);
  }, [selectionWindow]);

  useEffect(() => {
    if (!rehostOpeningId) return;
    setHoverCycle(null);
    lastCyclePickRef.current = null;
    setSelectionWindow(null);
    const cancelRehost = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      onCancelOpeningRehost();
    };
    window.addEventListener("keydown", cancelRehost, true);
    return () => window.removeEventListener("keydown", cancelRehost, true);
  }, [onCancelOpeningRehost, rehostOpeningId]);

  useEffect(() => {
    if (activeTool !== "object" || !placementFamily) return;
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
      const consume = () => {
        event.preventDefault();
        event.stopImmediatePropagation();
      };
      if (event.key === "Escape") {
        consume();
        onCancelFixturePlacement();
        return;
      }
      if (event.key.toLowerCase() === "r") {
        consume();
        rotatePlacement(event.shiftKey);
      }
    };
    window.addEventListener("keydown", handler, true);
    return () => window.removeEventListener("keydown", handler, true);
  }, [activeTool, onCancelFixturePlacement, placementFamily, rotatePlacement]);

  useEffect(() => {
    if (activeTool !== "select" || !hoverCycle?.candidates.length) return;
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
      if (event.key === "Escape") {
        setHoverCycle(null);
        return;
      }
      if (event.key === "Tab" && hoverCycle.candidates.length > 1) {
        event.preventDefault();
        event.stopImmediatePropagation();
        const index = cycleSelectionIndex(
          hoverCycle.index,
          hoverCycle.candidates.length,
          event.shiftKey,
        );
        const candidate = hoverCycle.candidates[index];
        setHoverCycle((current) => current ? { ...current, index } : current);
        setSnapLabel(`${index + 1}/${hoverCycle.candidates.length} · ${selectionCandidateLabel(graph, candidate)} · Enter selects`);
        return;
      }
      if (event.key === "Enter") {
        event.preventDefault();
        event.stopImmediatePropagation();
        const candidate = hoverCycle.candidates[hoverCycle.index];
        onSelect(candidate, event.ctrlKey || event.metaKey || event.shiftKey);
        setSnapLabel(`${selectionCandidateLabel(graph, candidate)} selected`);
      }
    };
    window.addEventListener("keydown", handler, true);
    return () => window.removeEventListener("keydown", handler, true);
  }, [activeTool, graph, hoverCycle, onSelect]);

  const beginPinch = () => {
    const pointers = [...touchPointersRef.current.entries()].slice(0, 2);
    if (pointers.length < 2) return;
    const [[firstId, first], [secondId, second]] = pointers;
    const center: [number, number] = [
      (first[0] + second[0]) / 2,
      (first[1] + second[1]) / 2,
    ];
    const anchorWorld = toRawWorld({ clientX: center[0], clientY: center[1] });
    if (!anchorWorld) return;
    if (drag) onCancelGesture();
    setDrag(null);
    setSelectionWindow(null);
    setHoverCycle(null);
    resetConstruction();
    pinchRef.current = {
      pointerIds: [firstId, secondId],
      startDistance: Math.max(1, Math.hypot(second[0] - first[0], second[1] - first[1])),
      startCenter: center,
      anchorWorld,
      startView: planView,
    };
    setIsPanning(true);
    svgRef.current?.setPointerCapture(firstId);
    svgRef.current?.setPointerCapture(secondId);
  };

  const onPlanPointerDownCapture = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (event.pointerType === "touch") {
      touchPointersRef.current.set(event.pointerId, [event.clientX, event.clientY]);
      if (touchPointersRef.current.size >= 2) {
        event.preventDefault();
        event.stopPropagation();
        beginPinch();
        return;
      }
    }
    const wantsPan = event.button === 1 || (spacePan && event.button === 0);
    if (wantsPan) {
      event.preventDefault();
      event.stopPropagation();
      if (drag) onCancelGesture();
      setDrag(null);
      setSelectionWindow(null);
      setHoverCycle(null);
      navigationDragRef.current = {
        pointerId: event.pointerId,
        lastClientX: event.clientX,
        lastClientY: event.clientY,
      };
      setIsPanning(true);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    if (activeTool === "object" && placementFamily && event.button === 0) {
      event.preventDefault();
      event.stopPropagation();
      const snap = toWorld(event);
      const placement = fixturePlacementAt(graph, placementFamily, levelId, snap.point, placementYaw);
      setPlacementCursor(snap.point);
      setGuides(snap.guides);
      setSnapLabel(fixturePlacementMessage(placement));
      if (placement.valid) onPlaceFixture(placement);
      return;
    }
    onConstructionPointerDown(event);
  };

  const updateNavigationPointer = (event: ReactPointerEvent<SVGSVGElement>): boolean => {
    if (event.pointerType === "touch") {
      touchPointersRef.current.set(event.pointerId, [event.clientX, event.clientY]);
    }
    const pinch = pinchRef.current;
    if (pinch) {
      const first = touchPointersRef.current.get(pinch.pointerIds[0]);
      const second = touchPointersRef.current.get(pinch.pointerIds[1]);
      if (!first || !second) return true;
      event.preventDefault();
      const center: [number, number] = [
        (first[0] + second[0]) / 2,
        (first[1] + second[1]) / 2,
      ];
      const distance = Math.max(1, Math.hypot(second[0] - first[0], second[1] - first[1]));
      const rect = svgRef.current?.getBoundingClientRect();
      const zoomed = zoomPlanViewAt(
        pinch.startView,
        pinch.anchorWorld,
        pinch.startDistance / distance,
        fittedView.width / 128,
        fittedView.width * 8,
      );
      setPlanView(panPlanViewByPixels(
        zoomed,
        center[0] - pinch.startCenter[0],
        center[1] - pinch.startCenter[1],
        rect?.width ?? 1,
        rect?.height ?? 1,
      ));
      return true;
    }
    const navigationDrag = navigationDragRef.current;
    if (!navigationDrag || navigationDrag.pointerId !== event.pointerId) return false;
    event.preventDefault();
    const rect = svgRef.current?.getBoundingClientRect();
    const deltaX = event.clientX - navigationDrag.lastClientX;
    const deltaY = event.clientY - navigationDrag.lastClientY;
    navigationDrag.lastClientX = event.clientX;
    navigationDrag.lastClientY = event.clientY;
    setPlanView((current) => panPlanViewByPixels(
      current,
      deltaX,
      deltaY,
      rect?.width ?? 1,
      rect?.height ?? 1,
    ));
    return true;
  };

  const finishNavigationPointer = (event: ReactPointerEvent<SVGSVGElement>): boolean => {
    const wasPinching = Boolean(pinchRef.current);
    if (event.pointerType === "touch") touchPointersRef.current.delete(event.pointerId);
    if (wasPinching && touchPointersRef.current.size < 2) {
      pinchRef.current = null;
      setIsPanning(false);
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      return true;
    }
    const navigationDrag = navigationDragRef.current;
    if (navigationDrag?.pointerId !== event.pointerId) return false;
    navigationDragRef.current = null;
    setIsPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    return true;
  };

  const onPointerMove = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (updateNavigationPointer(event)) return;
    if (selectionWindow) {
      const point = toRawWorld(event);
      if (!point) return;
      setSelectionWindow((current) => current ? { ...current, current: point } : current);
      return;
    }
    if (!drag) {
      if (activeTool === "object" && placementFamily) {
        const snap = toWorld(event);
        const placement = fixturePlacementAt(graph, placementFamily, levelId, snap.point, placementYaw);
        setPlacementCursor(snap.point);
        setGuides(snap.guides);
        setSnapLabel(fixturePlacementMessage(placement));
        setHoverCycle(null);
        return;
      }
      if (!rehostOpeningId) updateHoverPreselection(event);
      if (constructionStart && (activeTool === "wall" || activeTool === "measure")) {
        const snap = toWorld(event, constructionStart);
        setCursorPoint(snap.point);
        setGuides(snap.guides);
        setSnapLabel(snap.label);
      }
      return;
    }
    if (drag.mode === "fixture-rotate") {
      const fixture = graph.fixtures.find((item) => item.id === drag.selection.id);
      const point = toRawWorld(event);
      if (!fixture || !point) return;
      const rawYaw = Math.atan2(
        point[1] - fixture.center_m[1],
        point[0] - fixture.center_m[0],
      ) * 180 / Math.PI + 90;
      const step = event.altKey ? 0 : event.shiftKey ? 15 : 1;
      const yaw = step ? Math.round(rawYaw / step) * step : rawYaw;
      const normalized = ((yaw % 360) + 360) % 360;
      const validation = validateFixtureEntityChanges(graph, fixture.id, { yaw_deg: normalized });
      if (!validation.valid) {
        setSnapLabel(validation.reason ?? "Rotation blocked by component clearance");
        return;
      }
      const changes = validation.changesById[fixture.id] ?? { yaw_deg: normalized };
      const alignedYaw = Number(changes.yaw_deg ?? normalized);
      setSnapLabel(`${alignedYaw.toFixed(step ? 0 : 1)}° rotation${validation.notices.length ? ` · ${validation.notices[0]}` : ""}`);
      onPreviewGesture(drag.selection, changes);
      return;
    }
    const snap = toWorld(event, drag.origin);
    const point = snap.point;
    setGuides(snap.guides);
    setSnapLabel(snap.label);
    const entity = drag.selection.collection === "walls"
      ? graph.walls.find((item) => item.id === drag.selection.id)
      : drag.selection.collection === "rooms"
        ? graph.rooms.find((item) => item.id === drag.selection.id)
        : drag.selection.collection === "fixtures"
          ? graph.fixtures.find((item) => item.id === drag.selection.id)
          : drag.selection.collection === "openings"
            ? graph.openings.find((item) => item.id === drag.selection.id)
            : drag.selection.collection === "dimensions"
              ? (graph.dimensions ?? []).find((item) => item.id === drag.selection.id)
              : (graph.vertical_connections ?? []).find((item) => item.id === drag.selection.id);
    if (!entity) return;
    if (drag.mode === "measurement-from" || drag.mode === "measurement-to") {
      onPreviewGesture(
        drag.selection,
        drag.mode === "measurement-from" ? { from: point } : { to: point },
      );
      return;
    }
    if (drag.mode === "fixture") {
      const fixture = entity as FixtureEntity;
      const validation = validateFixtureEntityChanges(graph, fixture.id, { center_m: point });
      if (!validation.valid) {
        setSnapLabel(validation.reason ?? "Move blocked by component clearance");
        return;
      }
      onPreviewGesture(drag.selection, validation.changesById[fixture.id] ?? { center_m: point });
      setSnapLabel(`${snap.label || "Position"} · ${validation.placement ? fixturePlacementMessage(validation.placement) : "valid"}`);
      return;
    }
    if (drag.mode === "vertical") {
      onPreviewGesture(drag.selection, { center_m: point });
      return;
    }
    if (drag.mode === "room-vertex") {
      const room = entity as RoomEntity;
      const rect = svgRef.current?.getBoundingClientRect();
      const wallSnapTolerance = Math.max(
        planView.width / Math.max(1, rect?.width ?? 1),
        planView.height / Math.max(1, rect?.height ?? 1),
      ) * 12;
      const wallSnap = event.altKey
        ? null
        : snapRoomBoundaryPoint(
            point,
            graph.walls.filter((wall) => wall.level_id === levelId),
            wallSnapTolerance,
          );
      const boundaryPoint = wallSnap?.point ?? point;
      const result = movePolygonVertex(room.polygon, drag.vertexIndex ?? -1, boundaryPoint);
      if (!result.valid) {
        setSnapLabel(
          result.reason === "self_intersection"
            ? "Boundary crossing blocked"
            : "Room area too small",
        );
        return;
      }
      if (!onPreviewGesture(drag.selection, { polygon: result.polygon })) {
        setSnapLabel("Boundary change blocked by room or object relationships");
        return;
      }
      setSnapLabel(wallSnap ? `Wall axis · ${wallSnap.wallId}` : snap.label);
      return;
    }
    if (drag.mode === "opening" || drag.mode === "opening-start" || drag.mode === "opening-end") {
      const opening = entity as OpeningEntity;
      const host = graph.walls.find((item) => item.id === opening.wall_id);
      if (!host) return;
      if (drag.mode === "opening") {
        const placement = moveOpeningToPoint(opening, host, graph.openings, point);
        if (!placement.valid || !placement.changes) {
          setSnapLabel(openingPlacementMessage(placement.reason, placement.conflictId));
          return;
        }
        onPreviewGesture(drag.selection, placement.changes);
        return;
      }
      if (!drag.anchor) return;
      const resized = resizeOpeningFromEdge(point, drag.anchor, host);
      if (!resized.valid || !resized.changes) {
        setSnapLabel("Minimum opening width is 200 mm");
        return;
      }
      const placement = validateOpeningPlacement(
        { ...opening, ...resized.changes },
        host,
        graph.openings,
      );
      if (!placement.valid || !placement.changes) {
        setSnapLabel(openingPlacementMessage(placement.reason, placement.conflictId));
        return;
      }
      onPreviewGesture(drag.selection, placement.changes);
      return;
    }
    const wall = entity as WallEntity;
    onPreviewGesture(
      drag.selection,
      drag.mode === "wall-from" ? { from: point } : { to: point },
    );
  };

  const beginSelectionWindow = (
    event: ReactPointerEvent<SVGElement>,
    clickSelection?: Selection,
  ) => {
    const point = toRawWorld(event);
    if (!point) return;
    setSelectionWindow({
      start: point,
      current: point,
      additive: event.ctrlKey || event.metaKey || event.shiftKey,
      pointerId: event.pointerId,
      clickSelection,
    });
    setHoverCycle(null);
    svgRef.current?.setPointerCapture(event.pointerId);
  };

  const onCanvasPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (activeTool !== "select") return;
    const entity = (event.target as Element).closest("[data-collection][data-id]");
    if (!entity) beginSelectionWindow(event);
  };

  const onConstructionPointerDown = (event: ReactPointerEvent<SVGSVGElement>) => {
    if (activeTool !== "wall" && activeTool !== "measure") return;
    event.stopPropagation();
    const snap = toWorld(event, constructionStart ?? undefined);
    setGuides(snap.guides);
    setSnapLabel(snap.label);
    if (!constructionStart) {
      setConstructionStart(snap.point);
      setCursorPoint(snap.point);
      setPrecisionLength("");
      setPrecisionAngle("");
      setSnapLabel("Type a length, then Enter · Tab adds an angle");
      return;
    }
    completeConstruction(resolvePrecisionEndpoint(snap.point));
  };

  const roomItems = hidden.has("rooms") ? [] : graph.rooms.filter((item) => item.level_id === levelId && !isEntityHidden("rooms", item.id));
  const wallItems = hidden.has("walls") ? [] : graph.walls.filter((item) => item.level_id === levelId && !isEntityHidden("walls", item.id));
  const openingItems = hidden.has("openings") ? [] : graph.openings.filter((item) => item.level_id === levelId && !isEntityHidden("openings", item.id));
  const fixtureItems = hidden.has("fixtures") ? [] : graph.fixtures.filter((item) => item.level_id === levelId && !isEntityHidden("fixtures", item.id));
  const routeItems = hidden.has("routes") ? [] : graph.routes.filter((item) => item.level_id === levelId && !isEntityHidden("routes", item.id));
  const verticalItems = hidden.has("vertical_connections") ? [] : (graph.vertical_connections ?? []).filter(
    (item) => (item.from_level_id === levelId || item.to_level_id === levelId) && !isEntityHidden("vertical_connections", item.id),
  );
  const constraintItems = hidden.has("constraints") ? [] : (graph.constraints ?? []).filter(
    (item) => item.level_id === levelId && !isEntityHidden("constraints", item.id),
  );
  const dimensionItems = hidden.has("dimensions") ? [] : (graph.dimensions ?? []).filter(
    (item) => item.level_id === levelId && !isEntityHidden("dimensions", item.id),
  );
  const visibleSelections: Selection[] = [
    ...roomItems.map((item) => ({ collection: "rooms" as const, id: item.id })),
    ...wallItems.map((item) => ({ collection: "walls" as const, id: item.id })),
    ...openingItems.map((item) => ({ collection: "openings" as const, id: item.id })),
    ...fixtureItems.map((item) => ({ collection: "fixtures" as const, id: item.id })),
    ...routeItems.map((item) => ({ collection: "routes" as const, id: item.id })),
    ...verticalItems.map((item) => ({ collection: "vertical_connections" as const, id: item.id })),
    ...constraintItems.map((item) => ({ collection: "constraints" as const, id: item.id })),
    ...dimensionItems.map((item) => ({ collection: "dimensions" as const, id: item.id })),
  ];
  const selectableSelections = visibleSelections.filter(
    (item) => !selectionExclusionSet.has(item.collection),
  );

  function hitCandidatesAt(clientX: number, clientY: number) {
    const rect = svgRef.current?.getBoundingClientRect();
    const point = toRawWorld({ clientX, clientY });
    if (!rect || !point) return null;
    const worldPerPixel = Math.max(
      planView.width / Math.max(1, rect.width),
      planView.height / Math.max(1, rect.height),
    );
    return {
      point,
      candidates: hitTestPlanGraph(graph, selectableSelections, point, worldPerPixel * 9),
      cursor: [clientX - rect.left, clientY - rect.top] as [number, number],
      hud: [
        Math.min(Math.max(8, clientX - rect.left + 14), Math.max(8, rect.width - 184)),
        Math.min(Math.max(8, clientY - rect.top + 14), Math.max(8, rect.height - 62)),
      ] as [number, number],
    };
  }

  function updateHoverPreselection(event: ReactPointerEvent<SVGSVGElement>) {
    if (activeTool !== "select" || event.pointerType === "touch" || event.buttons !== 0) {
      setHoverCycle(null);
      return;
    }
    const hit = hitCandidatesAt(event.clientX, event.clientY);
    if (!hit?.candidates.length) {
      setHoverCycle(null);
      return;
    }
    const signature = hit.candidates.map(selectionKey).join("|");
    setHoverCycle((current) => ({
      ...hit,
      index: current && current.candidates.map(selectionKey).join("|") === signature
        ? Math.min(current.index, hit.candidates.length - 1)
        : 0,
    }));
  }

  function pickPointerSelection(
    fallback: Selection,
    event: ReactPointerEvent<SVGElement>,
  ): Selection {
    const hit = hitCandidatesAt(event.clientX, event.clientY);
    if (!hit || hit.candidates.length < 2) return fallback;
    const signature = hit.candidates.map(selectionKey).join("|");
    const hoverSignature = hoverCycle?.candidates.map(selectionKey).join("|");
    const fallbackIndex = Math.max(0, hit.candidates.findIndex(
      (candidate) => selectionKey(candidate) === selectionKey(fallback),
    ));
    let index = hoverSignature === signature ? hoverCycle?.index ?? fallbackIndex : fallbackIndex;
    const previous = lastCyclePickRef.current;
    const repeated = previous?.signature === signature
      && Math.hypot(event.clientX - previous.clientX, event.clientY - previous.clientY) <= 8
      && performance.now() - previous.timestamp <= 1800;
    if (repeated) index = cycleSelectionIndex(previous.index, hit.candidates.length);
    lastCyclePickRef.current = {
      signature,
      clientX: event.clientX,
      clientY: event.clientY,
      timestamp: performance.now(),
      index,
    };
    setHoverCycle({ ...hit, index });
    const picked = hit.candidates[index];
    setSnapLabel(`${index + 1}/${hit.candidates.length} · ${selectionCandidateLabel(graph, picked)} · click again or press Tab`);
    return picked;
  }

  const activeSelectionRectangle = selectionWindow
    ? selectionRectangle(selectionWindow.start, selectionWindow.current)
    : null;
  const activeSelectionMode = selectionWindow
    ? selectionMode(selectionWindow.start, selectionWindow.current)
    : null;
  const activeWindowMatches = activeSelectionRectangle && activeSelectionMode
    ? selectInRectangle(graph, selectableSelections, activeSelectionRectangle, activeSelectionMode)
    : [];
  const preselected = rehostOpeningId ? null : hoverCycle?.candidates[hoverCycle.index] ?? null;
  const preselectionFootprint = preselected ? elementFootprint(graph, preselected) : null;
  const viewportHint = rehostOpeningId
    ? "Pick a highlighted wall · click position is preferred · Escape cancels"
    : activeTool === "select"
    ? selection?.collection === "rooms"
      ? "Drag circle · Drag diamond to add · Right click circle to remove · Alt disables wall snap"
      : "Wheel zoom · Space pan · → Window · ← Crossing"
    : activeTool === "wall" || activeTool === "measure"
      ? "Shift locks orthogonal · Alt disables snap · Type length + angle"
      : activeTool === "object"
        ? "Click to place · R rotate · Shift+R reverse · Esc finish"
        : "Click the plan to place · Escape cancels";

  return (
    <div className="viewport plan-viewport" aria-label="2D plan editor">
      <div className="viewport-label">2D PLAN</div>
      <svg
        ref={svgRef}
        viewBox={viewBox}
        className={`tool-${activeTool}${rehostOpeningId ? " rehosting-opening" : ""}${spacePan ? " space-pan" : ""}${isPanning ? " is-panning" : ""}`}
        onPointerDownCapture={onPlanPointerDownCapture}
        onPointerDown={onCanvasPointerDown}
        onPointerMove={onPointerMove}
        onWheel={onPlanWheel}
        onAuxClick={(event) => event.button === 1 && event.preventDefault()}
        onPointerUp={(event) => {
          if (finishNavigationPointer(event)) return;
          if (selectionWindow) {
            const end = toRawWorld(event) ?? selectionWindow.current;
            const rect = svgRef.current?.getBoundingClientRect();
            const worldPerPixel = Math.max(
              planView.width / Math.max(1, rect?.width ?? 1),
              planView.height / Math.max(1, rect?.height ?? 1),
            );
            const moved = distanceMeters(selectionWindow.start, end);
            if (moved >= worldPerPixel * 4) {
              const rectangle = selectionRectangle(selectionWindow.start, end);
              const mode = selectionMode(selectionWindow.start, end);
              onSelectMany(
                selectInRectangle(graph, visibleSelections, rectangle, mode),
                selectionWindow.additive,
              );
            } else if (selectionWindow.clickSelection) {
              onSelect(selectionWindow.clickSelection, selectionWindow.additive);
            } else if (!selectionWindow.additive) {
              onClearSelection();
            }
            if (event.currentTarget.hasPointerCapture(selectionWindow.pointerId)) {
              event.currentTarget.releasePointerCapture(selectionWindow.pointerId);
            }
            setSelectionWindow(null);
            setSnapLabel("");
            return;
          }
          if (drag) onCommitGesture();
          setDrag(null);
          if (!constructionStart) setGuides([]);
        }}
        onPointerCancel={(event) => {
          if (finishNavigationPointer(event)) return;
          setSelectionWindow(null);
          if (drag) onCancelGesture();
          setDrag(null);
          setGuides([]);
        }}
        onPointerLeave={() => {
          if (!drag && !constructionStart && !isPanning) {
            setGuides([]);
            setCursorPoint(null);
            setPlacementCursor(null);
            setHoverCycle(null);
          }
        }}
        onContextMenu={(event) => {
          const target = (event.target as Element).closest<SVGElement>("[data-collection][data-id]");
          if (!target?.dataset.collection || !target.dataset.id) return;
          event.preventDefault();
          event.stopPropagation();
          onOpenContextMenu(
            {
              collection: target.dataset.collection as CollectionName,
              id: target.dataset.id,
            },
            event.clientX,
            event.clientY,
          );
        }}
      >
        {sourceUrl ? (
          <image
            className="source-drawing"
            href={sourceUrl}
            x={0}
            y={0}
            width={Math.max(bounds.maxX * 1.03, width)}
            height={Math.max(bounds.maxY * 1.03, height)}
            preserveAspectRatio="none"
          />
        ) : null}
        <g className={selectionExclusionSet.has("rooms") ? "room-layer selection-excluded" : "room-layer"}>
          {roomItems.map((room) => {
            const selected = isSelected("rooms", room.id);
            return (
              <polygon
                key={room.id}
                data-collection="rooms"
                data-id={room.id}
                points={room.polygon.map((point) => point.join(",")).join(" ")}
                className={selected ? "room selected" : "room"}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  beginSelectionWindow(
                    event,
                    pickPointerSelection({ collection: "rooms", id: room.id }, event),
                  );
                }}
              />
            );
          })}
        </g>
        <g className={selectionExclusionSet.has("walls") ? "wall-layer selection-excluded" : "wall-layer"}>
          {wallItems.map((wall) => {
            const selected = isSelected("walls", wall.id);
            const rehostOpening = rehostOpeningId
              ? graph.openings.find((item) => item.id === rehostOpeningId)
              : null;
            const hostCandidate = Boolean(rehostOpeningId);
            const currentHost = rehostOpening?.wall_id === wall.id;
            const hostLocked = isEntityLocked("walls", wall.id);
            return (
              <line
                key={wall.id}
                data-collection="walls"
                data-id={wall.id}
                x1={wall.from[0]}
                y1={wall.from[1]}
                x2={wall.to[0]}
                y2={wall.to[1]}
                strokeWidth={Math.max(wall.thickness_m, 0.08)}
                className={[
                  "wall",
                  selected ? "selected" : "",
                  hostCandidate ? "host-candidate" : "",
                  currentHost ? "current-host" : "",
                  hostLocked ? "host-locked" : "",
                ].filter(Boolean).join(" ")}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  if (rehostOpeningId) {
                    event.preventDefault();
                    const point = toRawWorld(event) ?? [
                      (wall.from[0] + wall.to[0]) / 2,
                      (wall.from[1] + wall.to[1]) / 2,
                    ] as [number, number];
                    onPickOpeningHost(wall.id, point);
                    return;
                  }
                  onSelect(
                    pickPointerSelection({ collection: "walls", id: wall.id }, event),
                    event.ctrlKey || event.metaKey,
                  );
                }}
              />
            );
          })}
        </g>
        <g className={selectionExclusionSet.has("constraints") ? "constraint-layer selection-excluded" : "constraint-layer"}>
          {constraintItems.map((constraint) => {
            const reference = constraint.references[0];
            const wall = reference
              ? graph.walls.find((item) => item.id === reference.entity_id)
              : null;
            const point = wall && reference ? wall[reference.handle] : null;
            if (!point) return null;
            const selected = isSelected("constraints", constraint.id);
            return (
              <g
                key={constraint.id}
                data-collection="constraints"
                data-id={constraint.id}
                className={selected ? "constraint selected" : "constraint"}
                transform={`translate(${point[0]} ${point[1]})`}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  onSelect(
                    pickPointerSelection({ collection: "constraints", id: constraint.id }, event),
                    event.ctrlKey || event.metaKey,
                  );
                }}
              >
                <circle cx={-0.055} cy={0} r={0.07} />
                <circle cx={0.055} cy={0} r={0.07} />
                <line x1={-0.015} y1={0} x2={0.015} y2={0} />
              </g>
            );
          })}
        </g>
        <g className={selectionExclusionSet.has("openings") ? "opening-layer selection-excluded" : "opening-layer"}>
          {openingItems.map((opening) => {
            const selected = isSelected("openings", opening.id);
            const host = graph.walls.find((item) => item.id === opening.wall_id);
            const frame = host ? openingFrame(opening, host) : null;
            if (!frame) return null;
            const isDoor = opening.type === "door";
            const showSwing = isDoor && opening.operation_type !== "sliding" && opening.operation_type !== "folding";
            return (
              <g
                key={opening.id}
                data-collection="openings"
                data-id={opening.id}
                className={selected ? "opening selected" : "opening"}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  const fallback = { collection: "openings", id: opening.id } as Selection;
                  const next = pickPointerSelection(fallback, event);
                  onSelect(next, event.ctrlKey || event.metaKey);
                  if (selectionKey(next) !== selectionKey(fallback)) return;
                  if (!onBeginGesture(next)) return;
                  setDrag({ selection: next, mode: "opening" });
                  event.currentTarget.setPointerCapture(event.pointerId);
                }}
              >
                <line
                  className="opening-cut"
                  x1={frame.start[0]}
                  y1={frame.start[1]}
                  x2={frame.end[0]}
                  y2={frame.end[1]}
                />
                {showSwing ? (
                  <>
                    <line className="door-leaf" x1={frame.hinge[0]} y1={frame.hinge[1]} x2={frame.openLeafEnd[0]} y2={frame.openLeafEnd[1]} />
                    <path className="door-arc" d={frame.arcPath} />
                    <circle className="door-hinge" cx={frame.hinge[0]} cy={frame.hinge[1]} r={0.055} />
                  </>
                ) : null}
                {!isDoor ? <line className="window-glass" x1={frame.start[0]} y1={frame.start[1]} x2={frame.end[0]} y2={frame.end[1]} /> : null}
                {selected && !isEntityLocked("openings", opening.id) ? (
                  <g className="opening-handles">
                    <circle
                      cx={frame.start[0]}
                      cy={frame.start[1]}
                      r={0.12}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        const next = { collection: "openings", id: opening.id } as Selection;
                        onSelect(next);
                        if (!onBeginGesture(next)) return;
                        setDrag({ selection: next, mode: "opening-start", anchor: frame.end });
                        event.currentTarget.setPointerCapture(event.pointerId);
                      }}
                    />
                    <circle
                      cx={frame.end[0]}
                      cy={frame.end[1]}
                      r={0.12}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        const next = { collection: "openings", id: opening.id } as Selection;
                        onSelect(next);
                        if (!onBeginGesture(next)) return;
                        setDrag({ selection: next, mode: "opening-end", anchor: frame.start });
                        event.currentTarget.setPointerCapture(event.pointerId);
                      }}
                    />
                  </g>
                ) : null}
              </g>
            );
          })}
        </g>
        <g className={selectionExclusionSet.has("fixtures") ? "fixture-layer selection-excluded" : "fixture-layer"}>
          {fixtureItems.map((fixture: FixtureEntity) => {
            const selected = isSelected("fixtures", fixture.id);
            return (
              <g
                key={fixture.id}
                data-collection="fixtures"
                data-id={fixture.id}
                transform={`translate(${fixture.center_m[0]} ${fixture.center_m[1]}) rotate(${fixture.yaw_deg ?? 0})`}
                className={selected ? "fixture selected" : "fixture"}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  const fallback = { collection: "fixtures", id: fixture.id } as Selection;
                  const next = pickPointerSelection(fallback, event);
                  onSelect(next, event.ctrlKey || event.metaKey);
                  if (selectionKey(next) !== selectionKey(fallback)) return;
                  if (!onBeginGesture(next)) return;
                  setDrag({ selection: next, mode: "fixture" });
                  event.currentTarget.setPointerCapture(event.pointerId);
                }}
              >
                <rect
                  x={-fixture.size_m[0] / 2}
                  y={-fixture.size_m[1] / 2}
                  width={fixture.size_m[0]}
                  height={fixture.size_m[1]}
                  rx={0.05}
                />
                {selected && !isEntityLocked("fixtures", fixture.id) ? (
                  <>
                    <line className="rotate-stem" x1={0} y1={-fixture.size_m[1] / 2} x2={0} y2={-fixture.size_m[1] / 2 - 0.25} />
                    <circle
                      className="rotate-handle"
                      cx={0}
                      cy={-fixture.size_m[1] / 2 - 0.25}
                      r={0.09}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        const next = { collection: "fixtures", id: fixture.id } as Selection;
                        onSelect(next);
                        if (!onBeginGesture(next)) return;
                        setDrag({
                          selection: next,
                          mode: "fixture-rotate",
                          origin: fixture.center_m,
                        });
                        event.currentTarget.setPointerCapture(event.pointerId);
                      }}
                    />
                  </>
                ) : null}
              </g>
            );
          })}
        </g>
        <g className={selectionExclusionSet.has("routes") ? "route-layer selection-excluded" : "route-layer"}>
          {routeItems.map((route) => {
            const selected = isSelected("routes", route.id);
            return (
              <polyline
                key={route.id}
                data-collection="routes"
                data-id={route.id}
                points={route.points_m.map(([x, y]) => `${x},${y}`).join(" ")}
                className={selected ? `route ${route.discipline ?? ""} selected` : `route ${route.discipline ?? ""}`}
                strokeWidth={Math.max(0.04, route.section_m?.[0] ?? 0.05)}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  onSelect(
                    pickPointerSelection({ collection: "routes", id: route.id }, event),
                    event.ctrlKey || event.metaKey,
                  );
                }}
              />
            );
          })}
        </g>
        <g className={selectionExclusionSet.has("vertical_connections") ? "vertical-layer selection-excluded" : "vertical-layer"}>
          {verticalItems.map((connection) => {
            const selected = isSelected("vertical_connections", connection.id);
            return (
              <rect
                key={connection.id}
                data-collection="vertical_connections"
                data-id={connection.id}
                x={connection.center_m[0] - connection.footprint_m[0] / 2}
                y={connection.center_m[1] - connection.footprint_m[1] / 2}
                width={connection.footprint_m[0]}
                height={connection.footprint_m[1]}
                transform={`rotate(${connection.yaw_deg ?? 0} ${connection.center_m[0]} ${connection.center_m[1]})`}
                className={selected ? "vertical-connection selected" : "vertical-connection"}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  const fallback = { collection: "vertical_connections", id: connection.id } as Selection;
                  const next = pickPointerSelection(fallback, event);
                  onSelect(next, event.ctrlKey || event.metaKey);
                  if (selectionKey(next) !== selectionKey(fallback)) return;
                  if (!onBeginGesture(next)) return;
                  setDrag({ selection: next, mode: "vertical" });
                  event.currentTarget.setPointerCapture(event.pointerId);
                }}
              />
            );
          })}
        </g>
        <g className={selectionExclusionSet.has("dimensions") ? "measurement-layer selection-excluded" : "measurement-layer"}>
          {dimensionItems.map((measurement) => {
            const midpoint: [number, number] = [
              (measurement.from[0] + measurement.to[0]) / 2,
              (measurement.from[1] + measurement.to[1]) / 2,
            ];
            return (
              <g
                key={measurement.id}
                data-collection="dimensions"
                data-id={measurement.id}
                className={isSelected("dimensions", measurement.id) ? "measurement selected" : "measurement"}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  onSelect(
                    pickPointerSelection({ collection: "dimensions", id: measurement.id }, event),
                    event.ctrlKey || event.metaKey,
                  );
                }}
              >
                <line x1={measurement.from[0]} y1={measurement.from[1]} x2={measurement.to[0]} y2={measurement.to[1]} />
                <text x={midpoint[0]} y={midpoint[1]}>{distanceMeters(measurement.from, measurement.to).toFixed(3)} m</text>
                {isSelected("dimensions", measurement.id) && !isEntityLocked("dimensions", measurement.id) ? (
                  <g className="measurement-handles">
                    <circle
                      cx={measurement.from[0]}
                      cy={measurement.from[1]}
                      r={0.11}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        const next = { collection: "dimensions", id: measurement.id } as Selection;
                        onSelect(next);
                        if (!onBeginGesture(next)) return;
                        setDrag({ selection: next, mode: "measurement-from", origin: measurement.to });
                        event.currentTarget.setPointerCapture(event.pointerId);
                      }}
                    />
                    <circle
                      cx={measurement.to[0]}
                      cy={measurement.to[1]}
                      r={0.11}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        const next = { collection: "dimensions", id: measurement.id } as Selection;
                        onSelect(next);
                        if (!onBeginGesture(next)) return;
                        setDrag({ selection: next, mode: "measurement-to", origin: measurement.from });
                        event.currentTarget.setPointerCapture(event.pointerId);
                      }}
                    />
                  </g>
                ) : (
                  <>
                    <circle cx={measurement.from[0]} cy={measurement.from[1]} r={0.05} />
                    <circle cx={measurement.to[0]} cy={measurement.to[1]} r={0.05} />
                  </>
                )}
              </g>
            );
          })}
        </g>
        {preselectionFootprint && preselected ? (
          <g
            className="preselection-layer"
            aria-label={`Preselected ${selectionCandidateLabel(graph, preselected)}`}
          >
            {preselectionFootprint.points.length === 1 ? (
              <circle
                cx={preselectionFootprint.points[0][0]}
                cy={preselectionFootprint.points[0][1]}
                r={Math.max(planView.width, planView.height) * 0.008}
              />
            ) : preselectionFootprint.closed ? (
              <polygon points={preselectionFootprint.points.map((point) => point.join(",")).join(" ")} />
            ) : (
              <polyline points={preselectionFootprint.points.map((point) => point.join(",")).join(" ")} />
            )}
          </g>
        ) : null}
        {placementFamily && fixturePlacement ? (
          <g
            className={`fixture-placement-preview ${fixturePlacement.valid ? "valid" : "invalid"}`}
            aria-label={`${placementFamily.name} placement ${fixturePlacement.valid ? "valid" : "blocked"}`}
          >
            <polygon points={fixturePlacement.footprint.map((point) => point.join(",")).join(" ")} />
            <line
              x1={fixturePlacement.center[0] - placementFamily.size_m[0] * 0.18}
              y1={fixturePlacement.center[1]}
              x2={fixturePlacement.center[0] + placementFamily.size_m[0] * 0.18}
              y2={fixturePlacement.center[1]}
            />
            <line
              x1={fixturePlacement.center[0]}
              y1={fixturePlacement.center[1] - placementFamily.size_m[1] * 0.18}
              x2={fixturePlacement.center[0]}
              y2={fixturePlacement.center[1] + placementFamily.size_m[1] * 0.18}
            />
            <text
              x={fixturePlacement.center[0]}
              y={fixturePlacement.center[1] - placementFamily.size_m[1] / 2 - Math.max(width, height) * 0.012}
              fontSize={Math.max(width, height) * 0.009}
            >
              {fixturePlacement.valid
                ? fixturePlacement.roomId ?? "READY"
                : fixturePlacementMessage(fixturePlacement)}
            </text>
          </g>
        ) : null}
        {constructionStart && precisionPreview ? (
          <g className={`construction-preview ${activeTool}`}>
            <line x1={constructionStart[0]} y1={constructionStart[1]} x2={precisionPreview[0]} y2={precisionPreview[1]} />
            <text x={(constructionStart[0] + precisionPreview[0]) / 2} y={(constructionStart[1] + precisionPreview[1]) / 2}>
              {distanceMeters(constructionStart, precisionPreview).toFixed(3)} m
            </text>
          </g>
        ) : null}
        {activeSelectionRectangle && activeSelectionMode ? (
          <g className={`selection-window ${activeSelectionMode}`} aria-label={`${activeSelectionMode} selection preview`}>
            <rect
              x={activeSelectionRectangle.minX}
              y={activeSelectionRectangle.minY}
              width={activeSelectionRectangle.maxX - activeSelectionRectangle.minX}
              height={activeSelectionRectangle.maxY - activeSelectionRectangle.minY}
            />
            <text
              x={activeSelectionRectangle.minX + Math.max(width, height) * 0.008}
              y={activeSelectionRectangle.minY + Math.max(width, height) * 0.025}
              fontSize={Math.max(width, height) * 0.012}
            >
              {activeSelectionMode.toUpperCase()} · {activeWindowMatches.length}
            </text>
          </g>
        ) : null}
        <g className="smart-guide-layer">
          {guides.map((guide) =>
            guide.axis === "x" ? (
              <line key={`${guide.axis}:${guide.value}`} x1={guide.value} y1={bounds.minY - padding} x2={guide.value} y2={bounds.maxY + padding} />
            ) : (
              <line key={`${guide.axis}:${guide.value}`} x1={bounds.minX - padding} y1={guide.value} x2={bounds.maxX + padding} y2={guide.value} />
            ),
          )}
        </g>
        {selection?.collection === "walls" && !selectionExclusionSet.has("walls") && !isEntityLocked("walls", selection.id) ? (
          <WallHandles
            wall={graph.walls.find((item) => item.id === selection.id)}
            onStart={(mode, event) => {
              const wall = graph.walls.find((item) => item.id === selection.id);
              const origin = mode === "wall-from" ? wall?.to : wall?.from;
              if (!onBeginGesture(selection)) return;
              setDrag({ selection, mode, origin });
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
          />
        ) : null}
        {selection?.collection === "rooms" && !selectionExclusionSet.has("rooms") && !isEntityLocked("rooms", selection.id) ? (
          <RoomHandles
            room={graph.rooms.find((item) => item.id === selection.id)}
            onStart={(vertexIndex, event) => {
              const room = graph.rooms.find((item) => item.id === selection.id);
              if (!onBeginGesture(selection)) return;
              setDrag({
                selection,
                mode: "room-vertex",
                vertexIndex,
                origin: room?.polygon[vertexIndex],
              });
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onInsert={(edgeIndex, event) => {
                const room = graph.rooms.find((item) => item.id === selection.id);
                if (!room) return;
                const inserted = insertRoomBoundaryVertex(room.polygon, edgeIndex);
                if (!inserted.valid || inserted.vertexIndex === undefined) {
                  setSnapLabel(inserted.reason ?? "Room vertex could not be inserted");
                  return;
                }
                if (!onBeginGesture(selection)) return;
                if (!onPreviewGesture(selection, { polygon: inserted.polygon })) {
                  onCancelGesture();
                  setSnapLabel("Boundary insertion blocked by room or object relationships");
                  return;
                }
                setDrag({
                  selection,
                  mode: "room-vertex",
                  vertexIndex: inserted.vertexIndex,
                  origin: inserted.polygon[inserted.vertexIndex],
                });
                setSnapLabel("Vertex inserted · drag to position");
                event.currentTarget.setPointerCapture(event.pointerId);
              }}
              onRemove={(vertexIndex) => {
                const room = graph.rooms.find((item) => item.id === selection.id);
                if (!room) return;
                const removed = removeRoomBoundaryVertex(room.polygon, vertexIndex);
                if (!removed.valid) {
                  setSnapLabel(removed.reason ?? "Room vertex could not be removed");
                  return;
                }
                if (!onBeginGesture(selection)) return;
                if (!onPreviewGesture(selection, { polygon: removed.polygon })) {
                  onCancelGesture();
                  setSnapLabel("Boundary removal blocked by room or object relationships");
                  return;
                }
                onCommitGesture();
                setSnapLabel("Vertex removed · one Undo step");
              }}
          />
        ) : null}
        {rehostOpeningId ? (
          <g className="host-pick-layer" aria-label="Available opening host walls">
            {wallItems.map((wall) => (
              <g key={`host-pick:${wall.id}`}>
                <line
                data-host-wall-id={wall.id}
                x1={wall.from[0]}
                y1={wall.from[1]}
                x2={wall.to[0]}
                y2={wall.to[1]}
                strokeWidth={14}
                className={isEntityLocked("walls", wall.id) ? "host-pick-target locked" : "host-pick-target"}
                onPointerDown={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  const point = toRawWorld(event) ?? [
                    (wall.from[0] + wall.to[0]) / 2,
                    (wall.from[1] + wall.to[1]) / 2,
                  ] as [number, number];
                  onPickOpeningHost(wall.id, point);
                }}
              >
                <title>{isEntityLocked("walls", wall.id) ? `${wall.id} is locked` : `Use ${wall.id} as host`}</title>
              </line>
              </g>
            ))}
          </g>
        ) : null}
      </svg>
      {rehostOpeningId ? (
        <div className="host-pick-screen-targets" aria-label="Keyboard-selectable opening host walls">
          {wallItems.map((wall) => {
            const midpoint: [number, number] = [
              (wall.from[0] + wall.to[0]) / 2,
              (wall.from[1] + wall.to[1]) / 2,
            ];
            return (
              <button
                key={`host-screen:${wall.id}`}
                type="button"
                disabled={isEntityLocked("walls", wall.id)}
                aria-label={isEntityLocked("walls", wall.id) ? `${wall.id} is locked` : `Use ${wall.id} as host`}
                style={{
                  left: `${((midpoint[0] - planView.x) / planView.width) * 100}%`,
                  top: `${((midpoint[1] - planView.y) / planView.height) * 100}%`,
                }}
                onPointerDown={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                }}
                onClick={(event) => {
                  event.preventDefault();
                  event.stopPropagation();
                  onPickOpeningHost(wall.id, midpoint);
                }}
              />
            );
          })}
        </div>
      ) : null}
      <div
        className={selections.length ? "plan-navigation with-selection" : "plan-navigation"}
        aria-label="Plan view controls"
      >
        <button onClick={() => zoomFromCenter(1.25)} aria-label="Zoom out" title="Zoom out">
          <Minus size={14} />
        </button>
        <output aria-label="Plan zoom level">{zoomPercent}%</output>
        <button onClick={() => zoomFromCenter(0.8)} aria-label="Zoom in" title="Zoom in">
          <Plus size={14} />
        </button>
        <i />
        <button
          onClick={fitSelection}
          disabled={!selections.length}
          aria-label="Fit selection"
          title="Fit selection (F)"
        >
          <Focus size={14} />
        </button>
        <button onClick={fitAll} aria-label="Fit all" title="Fit all (Home)">
          <Maximize2 size={14} />
        </button>
      </div>
      {hoverCycle && hoverCycle.candidates.length > 1 && preselected ? (
        <div
          className="selection-cycle-hud"
          aria-label="Selection candidates"
          style={{ left: hoverCycle.hud[0], top: hoverCycle.hud[1] }}
        >
          <span>{hoverCycle.index + 1}/{hoverCycle.candidates.length}</span>
          <div>
            <strong>{selectionCandidateLabel(graph, preselected)}</strong>
            <small>{preselected.collection.replaceAll("_", " ")} · {preselected.id}</small>
          </div>
          <kbd>Tab</kbd>
        </div>
      ) : null}
      {constructionStart ? (
        <form
          className="precision-hud"
          aria-label={`${activeTool === "wall" ? "Wall" : "Measurement"} precision input`}
          onSubmit={(event) => {
            event.preventDefault();
            completeConstruction(precisionPreview);
          }}
        >
          <div>
            <span>{activeTool === "wall" ? "WALL SEGMENT" : "MEASUREMENT"}</span>
            <strong>Exact input</strong>
          </div>
          <label>
            <span>Length</span>
            <input
              ref={lengthInputRef}
              value={precisionLength}
              inputMode="decimal"
              placeholder={`${currentPrecision.lengthM.toFixed(3)} m`}
              aria-invalid={Boolean(precisionLength.trim() && parseLengthInput(precisionLength) === null)}
              onChange={(event) => setPrecisionLength(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  event.preventDefault();
                  resetConstruction();
                }
              }}
            />
          </label>
          <label>
            <span>Angle</span>
            <input
              ref={angleInputRef}
              value={precisionAngle}
              inputMode="decimal"
              placeholder={`${currentPrecision.angleDeg.toFixed(1)} deg`}
              aria-invalid={Boolean(precisionAngle.trim() && parseAngleInput(precisionAngle) === null)}
              onChange={(event) => setPrecisionAngle(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Escape") {
                  event.preventDefault();
                  resetConstruction();
                }
              }}
            />
          </label>
          <button type="submit" disabled={!precisionValid}>
            {activeTool === "wall" ? "Place and continue" : "Place"}
          </button>
          <small>Type a value · Tab between fields · Enter places · Esc finishes</small>
        </form>
      ) : null}
      {placementFamily ? (
        <div className={`fixture-placement-hud${fixturePlacement && !fixturePlacement.valid ? " invalid" : ""}`} aria-label="Component placement controls">
          <span className={`family-glyph ${placementFamily.discipline}`}><Box size={17} /></span>
          <div>
            <span>PLACE COMPONENT · {placementFamily.mounting.toUpperCase()} HOST</span>
            <strong>{placementFamily.name}</strong>
            <small>
              {Math.round(placementFamily.size_m[0] * 1000)} × {Math.round(placementFamily.size_m[1] * 1000)} mm
              {fixturePlacement?.roomId ? ` · ${fixturePlacement.roomId}` : ""}
            </small>
          </div>
          <output>{fixturePlacement ? fixturePlacementMessage(fixturePlacement) : "Move over the plan"}</output>
          <button type="button" onClick={() => rotatePlacement(false)} title={placementFamily.mounting === "wall" ? "Flip on host wall (R)" : "Rotate 90 degrees (R)"}>
            <RotateCw size={14} /> <span>{placementFamily.mounting === "wall" ? "Flip" : "Rotate"}</span>
          </button>
          <button type="button" onClick={onChangeFixtureFamily} title="Choose another family">
            <Search size={14} /> <span>Change</span>
          </button>
          <button type="button" onClick={onCancelFixturePlacement} title="Finish placement (Escape)">
            <X size={14} /> <span>Finish</span>
          </button>
        </div>
      ) : null}
      {rehostOpeningId ? (
        <div className="opening-rehost-hud" aria-label="Pick a new opening host">
          <div>
            <span>PICK NEW HOST</span>
            <strong>{graph.openings.find((item) => item.id === rehostOpeningId)?.type ?? "Opening"}</strong>
            <small>{rehostOpeningId}</small>
          </div>
          <output>Choose a highlighted wall</output>
          <button type="button" onClick={onCancelOpeningRehost}>Cancel <kbd>Esc</kbd></button>
        </div>
      ) : null}
      {dimensionItems.length ? (
        <button
          className="clear-measurements"
          disabled={locked.has("dimensions") || dimensionItems.some((item) => isEntityLocked("dimensions", item.id))}
          onClick={onClearMeasurements}
          title={locked.has("dimensions") || dimensionItems.some((item) => isEntityLocked("dimensions", item.id)) ? "Unlock dimensions before clearing" : "Clear all dimensions"}
        >
          Clear dimensions
        </button>
      ) : null}
      <div className="snap-readout">{snapLabel || (activeTool === "object" ? "Move over the plan" : activeTool === "measure" ? "Pick two points" : activeTool === "wall" ? "Pick wall start" : "Ready")}</div>
      <div className="viewport-hint">{viewportHint}</div>
    </div>
  );
}

function openingPlacementMessage(reason?: string, conflictId?: string): string {
  if (reason === "overlap") return `Opening overlap blocked${conflictId ? ` · ${conflictId}` : ""}`;
  if (reason === "outside_wall") return "Opening must remain fully inside its host wall";
  if (reason === "too_narrow") return "Minimum opening width is 200 mm";
  return "Host wall geometry is invalid";
}

function WallHandles({
  wall,
  onStart,
}: {
  wall?: WallEntity;
  onStart: (
    mode: "wall-from" | "wall-to",
    event: ReactPointerEvent<SVGCircleElement>,
  ) => void;
}) {
  if (!wall) return null;
  return (
    <g className="wall-handles">
      <circle cx={wall.from[0]} cy={wall.from[1]} r={0.14} onPointerDown={(event) => onStart("wall-from", event)} />
      <circle cx={wall.to[0]} cy={wall.to[1]} r={0.14} onPointerDown={(event) => onStart("wall-to", event)} />
    </g>
  );
}

function RoomHandles({
  room,
  onStart,
  onInsert,
  onRemove,
}: {
  room?: RoomEntity;
  onStart: (vertexIndex: number, event: ReactPointerEvent<SVGCircleElement>) => void;
  onInsert: (edgeIndex: number, event: ReactPointerEvent<SVGRectElement>) => void;
  onRemove: (vertexIndex: number) => void;
}) {
  if (!room) return null;
  return (
    <g className="room-handles">
      {room.polygon.map((point, index) => {
        const next = room.polygon[(index + 1) % room.polygon.length];
        const midpoint: [number, number] = [
          (point[0] + next[0]) / 2,
          (point[1] + next[1]) / 2,
        ];
        return (
          <rect
            key={`${room.id}:edge:${index}`}
            className="room-midpoint-handle"
            x={midpoint[0] - 0.07}
            y={midpoint[1] - 0.07}
            width={0.14}
            height={0.14}
            transform={`rotate(45 ${midpoint[0]} ${midpoint[1]})`}
            onPointerDown={(event) => {
              event.stopPropagation();
              if (event.button !== 0) return;
              event.preventDefault();
              onInsert(index, event);
            }}
          >
            <title>Drag to insert a room boundary vertex</title>
          </rect>
        );
      })}
      {room.polygon.map((point, index) => (
        <circle
          key={`${room.id}:vertex:${index}`}
          cx={point[0]}
          cy={point[1]}
          r={0.13}
          onPointerDown={(event) => {
            event.stopPropagation();
            if (event.button !== 0) return;
            onStart(index, event);
          }}
          onContextMenu={(event) => {
            event.preventDefault();
            event.stopPropagation();
            onRemove(index);
          }}
        >
          <title>Drag vertex · right click to remove</title>
        </circle>
      ))}
    </g>
  );
}
