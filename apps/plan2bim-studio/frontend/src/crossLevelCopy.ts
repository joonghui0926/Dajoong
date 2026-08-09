import { collections, entities } from "./graph";
import type { BaseEntity, CollectionName, PlanGraph, Selection } from "./types";

export interface CrossLevelCopyItem {
  collection: CollectionName;
  entity: BaseEntity;
}

export interface CrossLevelCopyResult {
  items: CrossLevelCopyItem[];
  selections: Selection[];
  includedSourceSelections: Selection[];
  warnings: string[];
  conflicts: CrossLevelCopyConflict[];
}

export interface CrossLevelCopyConflict {
  collection: CollectionName;
  sourceId: string;
  targetId: string;
  reason: "coincident_wall" | "overlapping_room" | "overlapping_fixture" | "duplicate_route";
}

const copyOrder: CollectionName[] = [
  "rooms",
  "walls",
  "openings",
  "fixtures",
  "routes",
  "dimensions",
  "constraints",
];

const supportedCollections = new Set<CollectionName>(copyOrder);

function selectionKey(selection: Selection): string {
  return `${selection.collection}:${selection.id}`;
}

function sourceEntity(graph: PlanGraph, selection: Selection): BaseEntity | undefined {
  return entities(graph, selection.collection).find((entity) => entity.id === selection.id);
}

function cloneId(
  sourceId: string,
  collection: CollectionName,
  targetLevelId: string,
  usedIds: Set<string>,
): string {
  const parts = sourceId.split(":");
  const suffix = parts.length > 1 ? parts.slice(1).join(":") : `${collection}:${sourceId}`;
  const base = `${targetLevelId}:${suffix}:copy`;
  let candidate = base;
  let index = 2;
  while (usedIds.has(candidate)) candidate = `${base}:${index++}`;
  usedIds.add(candidate);
  return candidate;
}

function pointNear(left: unknown, right: unknown, tolerance = 0.005): boolean {
  if (!Array.isArray(left) || !Array.isArray(right) || left.length < 2 || right.length < 2) return false;
  return Math.hypot(Number(left[0]) - Number(right[0]), Number(left[1]) - Number(right[1])) <= tolerance;
}

function wallCoincides(source: BaseEntity, target: BaseEntity): boolean {
  return (
    (pointNear(source.from, target.from) && pointNear(source.to, target.to)) ||
    (pointNear(source.from, target.to) && pointNear(source.to, target.from))
  );
}

function polygonBounds(value: unknown): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length < 3) return null;
  const points = value.filter((point) => Array.isArray(point) && point.length >= 2) as number[][];
  if (points.length !== value.length) return null;
  const xs = points.map((point) => Number(point[0]));
  const ys = points.map((point) => Number(point[1]));
  if (![...xs, ...ys].every(Number.isFinite)) return null;
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function boundsOverlap(left: [number, number, number, number], right: [number, number, number, number], tolerance = 0.005): boolean {
  return (
    left[0] < right[2] - tolerance && left[2] > right[0] + tolerance &&
    left[1] < right[3] - tolerance && left[3] > right[1] + tolerance
  );
}

function fixtureBounds(entity: BaseEntity): [number, number, number, number] | null {
  if (!Array.isArray(entity.center_m) || !Array.isArray(entity.size_m)) return null;
  const [x, y] = entity.center_m.map(Number);
  const [width, depth] = entity.size_m.map(Number);
  if (![x, y, width, depth].every(Number.isFinite)) return null;
  return [x - width / 2, y - depth / 2, x + width / 2, y + depth / 2];
}

function routeMatches(source: BaseEntity, target: BaseEntity): boolean {
  if (!Array.isArray(source.points_m) || !Array.isArray(target.points_m)) return false;
  const sourcePoints = source.points_m as unknown[];
  const targetPoints = target.points_m as unknown[];
  return (
    sourcePoints.length === targetPoints.length &&
    sourcePoints.every((point, index) => {
      const other = targetPoints[index];
      return Array.isArray(point) && Array.isArray(other) &&
        point.length === other.length &&
        point.every((value, axis) => Math.abs(Number(value) - Number(other[axis])) <= 0.005);
    })
  );
}

function findCopyConflicts(
  graph: PlanGraph,
  selections: Selection[],
  targetLevelId: string,
): CrossLevelCopyConflict[] {
  const conflicts: CrossLevelCopyConflict[] = [];
  for (const selection of selections) {
    const source = sourceEntity(graph, selection);
    if (!source) continue;
    const targets = entities(graph, selection.collection).filter(
      (entity) => entity.level_id === targetLevelId,
    );
    for (const target of targets) {
      let reason: CrossLevelCopyConflict["reason"] | null = null;
      if (selection.collection === "walls" && wallCoincides(source, target)) {
        reason = "coincident_wall";
      } else if (selection.collection === "rooms") {
        const sourceBounds = polygonBounds(source.polygon);
        const targetBounds = polygonBounds(target.polygon);
        if (sourceBounds && targetBounds && boundsOverlap(sourceBounds, targetBounds)) reason = "overlapping_room";
      } else if (selection.collection === "fixtures") {
        const sourceBounds = fixtureBounds(source);
        const targetBounds = fixtureBounds(target);
        if (sourceBounds && targetBounds && boundsOverlap(sourceBounds, targetBounds)) reason = "overlapping_fixture";
      } else if (selection.collection === "routes" && routeMatches(source, target)) {
        reason = "duplicate_route";
      }
      if (reason) {
        conflicts.push({
          collection: selection.collection,
          sourceId: source.id,
          targetId: target.id,
          reason,
        });
        break;
      }
    }
  }
  return conflicts;
}

export function copySelectionsToLevel(
  graph: PlanGraph,
  initialSelections: Selection[],
  targetLevelId: string,
): CrossLevelCopyResult {
  if (!graph.levels.some((level) => level.id === targetLevelId)) {
    throw new Error(`Target level does not exist: ${targetLevelId}`);
  }
  const expanded = new Map<string, Selection>();
  const warnings: string[] = [];
  const include = (selection: Selection): boolean => {
    if (!supportedCollections.has(selection.collection)) {
      if (selection.collection === "vertical_connections") {
        warnings.push("Vertical connections require an explicit pair of levels and were not copied.");
      }
      return false;
    }
    if (!sourceEntity(graph, selection)) return false;
    const key = selectionKey(selection);
    if (expanded.has(key)) return false;
    expanded.set(key, selection);
    return true;
  };
  initialSelections.forEach(include);

  let changed = true;
  while (changed) {
    changed = false;
    for (const selection of [...expanded.values()]) {
      const entity = sourceEntity(graph, selection);
      if (!entity) continue;
      if (selection.collection === "openings" && typeof entity.wall_id === "string") {
        changed = include({ collection: "walls", id: entity.wall_id }) || changed;
      }
      if (selection.collection === "fixtures" && typeof entity.room_id === "string" && entity.room_id) {
        changed = include({ collection: "rooms", id: entity.room_id }) || changed;
      }
      if (selection.collection === "walls") {
        if (typeof entity.room_id === "string" && entity.room_id) {
          changed = include({ collection: "rooms", id: entity.room_id }) || changed;
        }
        for (const opening of graph.openings.filter((item) => item.wall_id === selection.id)) {
          changed = include({ collection: "openings", id: opening.id }) || changed;
        }
      }
      if (selection.collection === "constraints" && Array.isArray(entity.references)) {
        for (const reference of entity.references as Array<{ entity_id?: string }>) {
          if (reference.entity_id) {
            changed = include({ collection: "walls", id: reference.entity_id }) || changed;
          }
        }
      }
    }
  }

  const selectedWallIds = new Set(
    [...expanded.values()]
      .filter((selection) => selection.collection === "walls")
      .map((selection) => selection.id),
  );
  for (const constraint of graph.constraints ?? []) {
    if (
      constraint.references.length > 0 &&
      constraint.references.every((reference) => selectedWallIds.has(reference.entity_id))
    ) {
      include({ collection: "constraints", id: constraint.id });
    }
  }

  const conflicts = findCopyConflicts(graph, [...expanded.values()], targetLevelId);
  if (conflicts.length) {
    return {
      items: [],
      selections: [],
      includedSourceSelections: [...expanded.values()],
      warnings: [...new Set(warnings)],
      conflicts,
    };
  }

  const usedIds = new Set(
    collections.flatMap((collection) => entities(graph, collection).map((entity) => entity.id)),
  );
  const idMap = new Map<string, string>();
  for (const selection of expanded.values()) {
    idMap.set(
      selectionKey(selection),
      cloneId(selection.id, selection.collection, targetLevelId, usedIds),
    );
  }

  const items: CrossLevelCopyItem[] = [];
  for (const collection of copyOrder) {
    for (const selection of expanded.values()) {
      if (selection.collection !== collection) continue;
      const source = sourceEntity(graph, selection);
      const id = idMap.get(selectionKey(selection));
      if (!source || !id) continue;
      const clone = structuredClone(source);
      clone.id = id;
      clone.level_id = targetLevelId;
      clone.confidence = 1;
      clone.uncertainty = 0;
      clone.review_state = "accepted";
      clone.model_version = "human-correction";
      clone.copied_from_entity_id = source.id;
      delete clone.correction_id;
      delete clone.reviewed_by;
      if (collection === "walls" && typeof clone.room_id === "string" && clone.room_id) {
        clone.room_id = idMap.get(selectionKey({ collection: "rooms", id: clone.room_id })) ?? "";
      }
      if (collection === "openings") {
        clone.wall_id = idMap.get(selectionKey({ collection: "walls", id: String(clone.wall_id) })) ?? "";
      }
      if (collection === "fixtures" && typeof clone.room_id === "string" && clone.room_id) {
        clone.room_id = idMap.get(selectionKey({ collection: "rooms", id: clone.room_id })) ?? "";
      }
      if (collection === "constraints" && Array.isArray(clone.references)) {
        clone.references = (clone.references as Array<Record<string, unknown>>).map((reference) => ({
          ...reference,
          entity_id: idMap.get(selectionKey({
            collection: "walls",
            id: String(reference.entity_id ?? ""),
          })) ?? "",
        }));
      }
      items.push({ collection, entity: clone });
    }
  }
  return {
    items,
    selections: items.map((item) => ({ collection: item.collection, id: item.entity.id })),
    includedSourceSelections: [...expanded.values()],
    warnings: [...new Set(warnings)],
    conflicts: [],
  };
}
