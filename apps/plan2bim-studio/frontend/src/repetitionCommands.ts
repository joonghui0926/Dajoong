import { findEntity } from "./graph";
import type { EntityChanges } from "./editorCommands";
import type { BaseEntity, CollectionName, PlanGraph, Selection } from "./types";

export type MirrorAxis = "vertical" | "horizontal";

export interface PatternResult {
  valid: boolean;
  items: Array<{ collection: CollectionName; entity: BaseEntity }>;
  changesById: EntityChanges;
  reason?: string;
}

export function canPattern(selection: Selection): boolean {
  return selection.collection === "fixtures" || selection.collection === "vertical_connections";
}

export function patternCenter(graph: PlanGraph, selections: Selection[]): [number, number] | null {
  const centers = selections
    .map((selection) => centerOf(findEntity(graph, selection)))
    .filter((center): center is [number, number] => center !== null);
  if (centers.length !== selections.length || centers.length === 0) return null;
  return [
    centers.reduce((sum, center) => sum + center[0], 0) / centers.length,
    centers.reduce((sum, center) => sum + center[1], 0) / centers.length,
  ];
}

export function defaultMirrorCoordinates(
  graph: PlanGraph,
  selections: Selection[],
  clearanceM = 0.25,
): [number, number] | null {
  const positioned = selections.map((selection) => {
    const entity = findEntity(graph, selection);
    const center = centerOf(entity);
    if (!entity || !center) return null;
    const footprint = Array.isArray(entity.size_m)
      ? [Number(entity.size_m[0]), Number(entity.size_m[1])]
      : Array.isArray(entity.footprint_m)
        ? [Number(entity.footprint_m[0]), Number(entity.footprint_m[1])]
        : [0, 0];
    const yaw = THREE_DEGREES * (Number(entity.yaw_deg) || 0);
    const halfX = (Math.abs(Math.cos(yaw)) * footprint[0] + Math.abs(Math.sin(yaw)) * footprint[1]) / 2;
    const halfY = (Math.abs(Math.sin(yaw)) * footprint[0] + Math.abs(Math.cos(yaw)) * footprint[1]) / 2;
    return { center, halfX, halfY };
  });
  if (!positioned.length || positioned.some((item) => item === null)) return null;
  const valid = positioned.filter((item): item is NonNullable<typeof item> => item !== null);
  return [
    roundMetric(Math.max(...valid.map((item) => item.center[0] + item.halfX)) + clearanceM),
    roundMetric(Math.max(...valid.map((item) => item.center[1] + item.halfY)) + clearanceM),
  ];
}

export function mirrorPattern(
  graph: PlanGraph,
  selections: Selection[],
  axis: MirrorAxis,
  coordinateM: number,
  keepOriginal: boolean,
): PatternResult {
  const error = validatePatternSelection(graph, selections);
  if (error) return invalid(error);
  if (!Number.isFinite(coordinateM)) return invalid("Mirror coordinate must be finite.");
  const changesById: EntityChanges = {};
  const items: PatternResult["items"] = [];
  const reserved = existingIds(graph);
  for (const selection of selections) {
    const entity = findEntity(graph, selection);
    const center = centerOf(entity);
    if (!entity || !center) return invalid(`${selection.id} has no editable center.`);
    const yaw = Number.isFinite(Number(entity.yaw_deg)) ? Number(entity.yaw_deg) : 0;
    const changes = axis === "vertical"
      ? { center_m: [roundMetric(2 * coordinateM - center[0]), center[1]], yaw_deg: normalizeDegrees(180 - yaw) }
      : { center_m: [center[0], roundMetric(2 * coordinateM - center[1])], yaw_deg: normalizeDegrees(-yaw) };
    if (!keepOriginal) {
      changesById[entity.id] = changes;
      continue;
    }
    const clone = reviewedClone(entity, nextPatternId(entity.id, selection.collection, reserved));
    Object.assign(clone, changes);
    items.push({ collection: selection.collection, entity: clone });
  }
  return { valid: true, items, changesById };
}

export function linearArrayPattern(
  graph: PlanGraph,
  selections: Selection[],
  count: number,
  step: [number, number],
): PatternResult {
  const error = validatePatternSelection(graph, selections);
  if (error) return invalid(error);
  if (!Number.isInteger(count) || count < 2 || count > 100) {
    return invalid("Total instances must be an integer from 2 to 100.");
  }
  if (!step.every(Number.isFinite) || Math.hypot(...step) < 0.000001) {
    return invalid("Array spacing must contain a non-zero finite offset.");
  }
  if ((count - 1) * selections.length > 500) {
    return invalid("This array would create more than 500 objects in one command.");
  }
  const items: PatternResult["items"] = [];
  const reserved = existingIds(graph);
  for (let instance = 1; instance < count; instance += 1) {
    for (const selection of selections) {
      const entity = findEntity(graph, selection);
      const center = centerOf(entity);
      if (!entity || !center) return invalid(`${selection.id} has no editable center.`);
      const clone = reviewedClone(entity, nextPatternId(entity.id, selection.collection, reserved));
      clone.center_m = [
        roundMetric(center[0] + step[0] * instance),
        roundMetric(center[1] + step[1] * instance),
      ];
      items.push({ collection: selection.collection, entity: clone });
    }
  }
  return { valid: true, items, changesById: {} };
}

function validatePatternSelection(graph: PlanGraph, selections: Selection[]): string | null {
  if (!selections.length) return "Select at least one placed object.";
  const unsupported = selections.find((selection) => !canPattern(selection));
  if (unsupported) return `${unsupported.collection.replaceAll("_", " ")} cannot be patterned safely.`;
  const missing = selections.find((selection) => !findEntity(graph, selection));
  return missing ? `${missing.id} is no longer present in the graph.` : null;
}

function centerOf(entity: BaseEntity | null): [number, number] | null {
  if (!entity || !Array.isArray(entity.center_m) || entity.center_m.length < 2) return null;
  const center = entity.center_m.map(Number);
  return center.every(Number.isFinite) ? [center[0], center[1]] : null;
}

function existingIds(graph: PlanGraph): Map<CollectionName, Set<string>> {
  const collections: CollectionName[] = ["fixtures", "vertical_connections"];
  return new Map(collections.map((collection) => [
    collection,
    new Set(((graph[collection] as BaseEntity[] | undefined) ?? []).map((entity) => entity.id)),
  ]));
}

function nextPatternId(
  sourceId: string,
  collection: CollectionName,
  reserved: Map<CollectionName, Set<string>>,
): string {
  const ids = reserved.get(collection) ?? new Set<string>();
  reserved.set(collection, ids);
  let index = 1;
  let candidate = `${sourceId}:pattern:${index}`;
  while (ids.has(candidate)) candidate = `${sourceId}:pattern:${++index}`;
  ids.add(candidate);
  return candidate;
}

function reviewedClone(entity: BaseEntity, id: string): BaseEntity {
  return {
    ...structuredClone(entity),
    id,
    confidence: 1,
    uncertainty: 0,
    review_state: "accepted",
    model_version: "human-correction",
  };
}

function normalizeDegrees(value: number): number {
  const normalized = ((value % 360) + 360) % 360;
  return roundMetric(normalized > 180 ? normalized - 360 : normalized);
}

function roundMetric(value: number): number {
  const rounded = Math.round(value * 1_000_000_000) / 1_000_000_000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

function invalid(reason: string): PatternResult {
  return { valid: false, items: [], changesById: {}, reason };
}

const THREE_DEGREES = Math.PI / 180;
