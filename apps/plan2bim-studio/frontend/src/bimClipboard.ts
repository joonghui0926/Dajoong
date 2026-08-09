import { validateNewFixtures } from "./fixturePlacement";
import { collections, entities } from "./graph";
import { findNearestOpeningPlacement, validateOpeningPlacement } from "./openingGeometry";
import type {
  BaseEntity,
  CollectionName,
  FixtureEntity,
  OpeningEntity,
  PlanGraph,
  Selection,
  WallEntity,
} from "./types";

export interface BimClipboardItem {
  collection: CollectionName;
  entity: BaseEntity;
}

export interface BimClipboardBundle {
  schema_version: "dajoong.bim-clipboard.v1";
  source_project_id?: string;
  source_level_ids: string[];
  source_selections: Selection[];
  included_selections: Selection[];
  items: BimClipboardItem[];
  warnings: string[];
}

export interface BimPastePlan {
  valid: boolean;
  items: BimClipboardItem[];
  selections: Selection[];
  offset_m?: [number, number];
  warnings: string[];
  reason?: string;
}

const copyOrder: CollectionName[] = [
  "rooms",
  "walls",
  "openings",
  "fixtures",
  "routes",
  "vertical_connections",
  "dimensions",
  "constraints",
];
const supported = new Set(copyOrder);

export function createBimClipboardBundle(
  graph: PlanGraph,
  initialSelections: Selection[],
): BimClipboardBundle {
  const expanded = new Map<string, Selection>();
  const warnings: string[] = [];
  const include = (selection: Selection): boolean => {
    if (!supported.has(selection.collection)) {
      warnings.push(`${selection.collection.replaceAll("_", " ")} cannot be placed from the BIM clipboard.`);
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
      if (selection.collection === "walls") {
        for (const opening of graph.openings.filter((item) => item.wall_id === selection.id)) {
          changed = include({ collection: "openings", id: opening.id }) || changed;
        }
        for (const fixture of graph.fixtures.filter((item) => item.host_wall_id === selection.id)) {
          changed = include({ collection: "fixtures", id: fixture.id }) || changed;
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
    [...expanded.values()].filter((item) => item.collection === "walls").map((item) => item.id),
  );
  for (const constraint of graph.constraints ?? []) {
    if (constraint.references.length > 0
      && constraint.references.every((reference) => selectedWallIds.has(reference.entity_id))) {
      include({ collection: "constraints", id: constraint.id });
    }
  }

  const items = copyOrder.flatMap((collection) => [...expanded.values()]
    .filter((selection) => selection.collection === collection)
    .flatMap((selection) => {
      const entity = sourceEntity(graph, selection);
      return entity ? [{ collection, entity: structuredClone(entity) }] : [];
    }));
  const sourceLevelIds = [...new Set(items.flatMap((item) => entityLevels(item.entity)))];
  return {
    schema_version: "dajoong.bim-clipboard.v1",
    source_project_id: graph.project_id,
    source_level_ids: sourceLevelIds,
    source_selections: structuredClone(initialSelections),
    included_selections: [...expanded.values()],
    items,
    warnings: unique(warnings),
  };
}

export function planBimPaste(
  graph: PlanGraph,
  bundle: BimClipboardBundle,
  targetLevelId: string,
  snapIncrementM = 0.05,
): BimPastePlan {
  if (!bundle.items.length) return invalid("The BIM clipboard is empty.", bundle.warnings);
  if (!graph.levels.some((level) => level.id === targetLevelId)) {
    return invalid(`Target level does not exist: ${targetLevelId}`, bundle.warnings);
  }
  const warnings = [...bundle.warnings];
  const usedIds = new Set(collections.flatMap((collection) =>
    entities(graph, collection).map((entity) => entity.id),
  ));
  const idMap = new Map<string, string>();
  for (const item of bundle.items) {
    idMap.set(selectionKey({ collection: item.collection, id: item.entity.id }),
      pasteId(item.entity.id, item.collection, usedIds));
  }
  const sameLevel = bundle.source_level_ids.length === 1 && bundle.source_level_ids[0] === targetLevelId;
  const spacing = pasteSpacing(bundle.items, snapIncrementM);
  const offsets: Array<[number, number]> = sameLevel ? [] : [[0, 0]];
  const directions: Array<[number, number]> = [
    [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1],
  ];
  for (let ring = 1; ring <= 8; ring += 1) {
    for (const direction of directions) {
      offsets.push([direction[0] * spacing * ring, direction[1] * spacing * ring]);
    }
  }

  let lastReason = "No clear paste location was found on this level.";
  for (const offset of offsets) {
    const candidate = cloneBundle(graph, bundle, idMap, targetLevelId, offset);
    if (!candidate.valid) {
      lastReason = candidate.reason ?? lastReason;
      continue;
    }
    const validated = validateCandidate(graph, candidate.items);
    if (!validated.valid) {
      lastReason = validated.reason ?? lastReason;
      continue;
    }
    return {
      valid: true,
      items: validated.items,
      selections: validated.items.map((item) => ({ collection: item.collection, id: item.entity.id })),
      offset_m: offset,
      warnings: unique([...warnings, ...validated.warnings]),
    };
  }
  return invalid(lastReason, warnings);
}

function cloneBundle(
  graph: PlanGraph,
  bundle: BimClipboardBundle,
  idMap: Map<string, string>,
  targetLevelId: string,
  offset: [number, number],
): { valid: boolean; items: BimClipboardItem[]; reason?: string } {
  const items: BimClipboardItem[] = [];
  for (const item of bundle.items) {
    const clone = structuredClone(item.entity);
    const nextId = idMap.get(selectionKey({ collection: item.collection, id: clone.id }));
    if (!nextId) return { valid: false, items: [], reason: `No paste ID was allocated for ${clone.id}.` };
    const sourceId = clone.id;
    clone.id = nextId;
    clone.confidence = 1;
    clone.uncertainty = 0;
    clone.review_state = "accepted";
    clone.model_version = "human-correction";
    clone.pasted_from_entity_id = sourceId;
    delete clone.correction_id;
    delete clone.reviewed_by;

    if (item.collection === "vertical_connections") {
      const fromLevel = String(clone.from_level_id ?? "");
      const toLevel = String(clone.to_level_id ?? "");
      if (!graph.levels.some((level) => level.id === fromLevel)
        || !graph.levels.some((level) => level.id === toLevel)) {
        return { valid: false, items: [], reason: `${sourceId} references unavailable connected levels.` };
      }
    } else {
      clone.level_id = targetLevelId;
    }

    translateEntity(clone, item.collection, offset);
    if (item.collection === "openings") {
      const remapped = idMap.get(selectionKey({ collection: "walls", id: String(clone.wall_id ?? "") }));
      if (remapped) clone.wall_id = remapped;
      else {
        const host = graph.walls.find((wall) => wall.id === clone.wall_id);
        if (!host || host.level_id !== targetLevelId) {
          return { valid: false, items: [], reason: `${sourceId} requires its host wall on ${targetLevelId}.` };
        }
      }
    }
    if (item.collection === "fixtures") {
      const roomId = typeof clone.room_id === "string" ? clone.room_id : "";
      const hostWallId = typeof clone.host_wall_id === "string" ? clone.host_wall_id : "";
      if (roomId) clone.room_id = idMap.get(selectionKey({ collection: "rooms", id: roomId })) ?? roomId;
      if (hostWallId) clone.host_wall_id = idMap.get(selectionKey({ collection: "walls", id: hostWallId })) ?? hostWallId;
    }
    if (item.collection === "constraints" && Array.isArray(clone.references)) {
      clone.references = (clone.references as Array<Record<string, unknown>>).map((reference) => ({
        ...reference,
        entity_id: idMap.get(selectionKey({
          collection: "walls",
          id: String(reference.entity_id ?? ""),
        })) ?? reference.entity_id,
      }));
    }
    items.push({ collection: item.collection, entity: clone });
  }
  return { valid: true, items };
}

function validateCandidate(
  graph: PlanGraph,
  candidateItems: BimClipboardItem[],
): { valid: boolean; items: BimClipboardItem[]; warnings: string[]; reason?: string } {
  const walls = candidateItems.filter((item) => item.collection === "walls").map((item) => item.entity as WallEntity);
  for (const wall of walls) {
    const collision = graph.walls.find((existing) =>
      existing.level_id === wall.level_id && wallRectanglesOverlap(wall, existing),
    );
    if (collision) return { valid: false, items: [], warnings: [], reason: `${wall.id} overlaps ${collision.id}.` };
  }
  const rooms = candidateItems.filter((item) => item.collection === "rooms").map((item) => item.entity);
  for (const room of rooms) {
    const bounds = polygonBounds(room.polygon);
    const collision = bounds && graph.rooms.find((existing) => {
      if (existing.level_id !== room.level_id) return false;
      const existingBounds = polygonBounds(existing.polygon);
      return existingBounds && boundsOverlap(bounds, existingBounds);
    });
    if (collision) return { valid: false, items: [], warnings: [], reason: `${room.id} overlaps ${collision.id}.` };
  }

  const structuralGraph: PlanGraph = {
    ...graph,
    rooms: [...graph.rooms, ...(rooms as PlanGraph["rooms"])],
    walls: [...graph.walls, ...walls],
  };
  const normalizedOpenings: OpeningEntity[] = [];
  const openingItems = candidateItems.filter((item) => item.collection === "openings");
  for (const item of openingItems) {
    const opening = item.entity as OpeningEntity;
    const host = structuralGraph.walls.find((wall) => wall.id === opening.wall_id);
    if (!host) return { valid: false, items: [], warnings: [], reason: `${opening.id} has no valid host wall.` };
    const siblings = [...graph.openings, ...normalizedOpenings];
    const hostWasPasted = walls.some((wall) => wall.id === host.id);
    const placement = hostWasPasted
      ? validateOpeningPlacement(opening, host, siblings)
      : findNearestOpeningPlacement(opening, host, siblings, opening.center_m);
    if (!placement.valid || !placement.changes) {
      return { valid: false, items: [], warnings: [], reason: `${opening.id} has no collision-free host span.` };
    }
    normalizedOpenings.push({ ...opening, ...placement.changes });
  }

  const fixtures = candidateItems
    .filter((item) => item.collection === "fixtures")
    .map((item) => item.entity as FixtureEntity);
  const fixtureValidation = validateNewFixtures(structuralGraph, fixtures);
  if (!fixtureValidation.valid) {
    return { valid: false, items: [], warnings: [], reason: fixtureValidation.reason };
  }
  const openingMap = new Map(normalizedOpenings.map((opening) => [opening.id, opening]));
  const fixtureMap = new Map(fixtureValidation.fixtures.map((fixture) => [fixture.id, fixture]));
  return {
    valid: true,
    items: candidateItems.map((item) => {
      if (item.collection === "openings") return { ...item, entity: openingMap.get(item.entity.id) ?? item.entity };
      if (item.collection === "fixtures") return { ...item, entity: fixtureMap.get(item.entity.id) ?? item.entity };
      return item;
    }),
    warnings: fixtureValidation.notices,
  };
}

function translateEntity(entity: BaseEntity, collection: CollectionName, offset: [number, number]): void {
  if (collection === "walls") {
    entity.from = translatePoint(entity.from, offset);
    entity.to = translatePoint(entity.to, offset);
  } else if (collection === "rooms") {
    entity.polygon = translatePolygon(entity.polygon, offset);
  } else if (["openings", "fixtures", "vertical_connections"].includes(collection)) {
    entity.center_m = translatePoint(entity.center_m, offset);
  } else if (collection === "routes" && Array.isArray(entity.points_m)) {
    entity.points_m = (entity.points_m as unknown[]).map((point) => {
      if (!Array.isArray(point) || point.length < 2) return point;
      return [Number(point[0]) + offset[0], Number(point[1]) + offset[1], ...point.slice(2)];
    });
  } else if (collection === "dimensions") {
    entity.from = translatePoint(entity.from, offset);
    entity.to = translatePoint(entity.to, offset);
  }
}

function pasteSpacing(items: BimClipboardItem[], snap: number): number {
  const bounds = bundleBounds(items);
  const span = bounds ? Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]) : 0;
  return Math.max(Math.max(0.01, snap), span + Math.max(0.01, snap));
}

function bundleBounds(items: BimClipboardItem[]): [number, number, number, number] | null {
  const points: Array<[number, number]> = [];
  for (const item of items) {
    const entity = item.entity;
    if (item.collection === "walls") addPoints(points, entity.from, entity.to);
    else if (item.collection === "rooms" && Array.isArray(entity.polygon)) addPoints(points, ...(entity.polygon as unknown[]));
    else if (["openings", "fixtures", "vertical_connections"].includes(item.collection)) {
      const center = point(entity.center_m);
      if (!center) continue;
      const size = Array.isArray(entity.size_m) ? entity.size_m : entity.footprint_m;
      const width = Array.isArray(size) ? Number(size[0] ?? 0) : Number(entity.width_m ?? 0);
      const depth = Array.isArray(size) ? Number(size[1] ?? 0) : 0;
      points.push([center[0] - width / 2, center[1] - depth / 2], [center[0] + width / 2, center[1] + depth / 2]);
    } else if (item.collection === "routes" && Array.isArray(entity.points_m)) addPoints(points, ...(entity.points_m as unknown[]));
    else if (item.collection === "dimensions") addPoints(points, entity.from, entity.to);
  }
  if (!points.length) return null;
  return [
    Math.min(...points.map((value) => value[0])),
    Math.min(...points.map((value) => value[1])),
    Math.max(...points.map((value) => value[0])),
    Math.max(...points.map((value) => value[1])),
  ];
}

function wallRectanglesOverlap(left: WallEntity, right: WallEntity): boolean {
  const a = wallRectangle(left);
  const b = wallRectangle(right);
  if (!a || !b) return false;
  for (const polygon of [a, b]) {
    for (let index = 0; index < polygon.length; index += 1) {
      const start = polygon[index];
      const end = polygon[(index + 1) % polygon.length];
      const axis: [number, number] = [-(end[1] - start[1]), end[0] - start[0]];
      const leftProjection = project(a, axis);
      const rightProjection = project(b, axis);
      if (leftProjection.max <= rightProjection.min + 0.001 || rightProjection.max <= leftProjection.min + 0.001) return false;
    }
  }
  return true;
}

function wallRectangle(wall: WallEntity): Array<[number, number]> | null {
  const dx = wall.to[0] - wall.from[0];
  const dy = wall.to[1] - wall.from[1];
  const length = Math.hypot(dx, dy);
  if (length <= 0.01) return null;
  const half = Math.max(0.02, Number(wall.thickness_m)) / 2;
  const normal: [number, number] = [-dy / length * half, dx / length * half];
  return [
    [wall.from[0] + normal[0], wall.from[1] + normal[1]],
    [wall.to[0] + normal[0], wall.to[1] + normal[1]],
    [wall.to[0] - normal[0], wall.to[1] - normal[1]],
    [wall.from[0] - normal[0], wall.from[1] - normal[1]],
  ];
}

function project(polygon: Array<[number, number]>, axis: [number, number]): { min: number; max: number } {
  const values = polygon.map((value) => value[0] * axis[0] + value[1] * axis[1]);
  return { min: Math.min(...values), max: Math.max(...values) };
}

function polygonBounds(value: unknown): [number, number, number, number] | null {
  if (!Array.isArray(value) || value.length < 3) return null;
  const points = value.map(point);
  if (points.some((item) => !item)) return null;
  const valid = points as Array<[number, number]>;
  return [
    Math.min(...valid.map((item) => item[0])), Math.min(...valid.map((item) => item[1])),
    Math.max(...valid.map((item) => item[0])), Math.max(...valid.map((item) => item[1])),
  ];
}

function boundsOverlap(left: [number, number, number, number], right: [number, number, number, number]): boolean {
  return left[0] < right[2] - 0.005 && left[2] > right[0] + 0.005
    && left[1] < right[3] - 0.005 && left[3] > right[1] + 0.005;
}

function translatePoint(value: unknown, offset: [number, number]): [number, number] {
  const candidate = point(value) ?? [0, 0];
  return [round(candidate[0] + offset[0]), round(candidate[1] + offset[1])];
}

function translatePolygon(value: unknown, offset: [number, number]): Array<[number, number]> {
  if (!Array.isArray(value)) return [];
  return value.map((item) => translatePoint(item, offset));
}

function addPoints(target: Array<[number, number]>, ...values: unknown[]): void {
  for (const value of values) {
    const candidate = point(value);
    if (candidate) target.push(candidate);
  }
}

function point(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const candidate: [number, number] = [Number(value[0]), Number(value[1])];
  return candidate.every(Number.isFinite) ? candidate : null;
}

function entityLevels(entity: BaseEntity): string[] {
  return [entity.level_id, entity.from_level_id, entity.to_level_id]
    .filter((value): value is string => typeof value === "string" && value.length > 0);
}

function sourceEntity(graph: PlanGraph, selection: Selection): BaseEntity | undefined {
  return entities(graph, selection.collection).find((entity) => entity.id === selection.id);
}

function selectionKey(selection: Selection): string {
  return `${selection.collection}:${selection.id}`;
}

function pasteId(sourceId: string, collection: CollectionName, usedIds: Set<string>): string {
  const base = `${sourceId}:paste` || `${collection}:paste`;
  let candidate = base;
  let index = 2;
  while (usedIds.has(candidate)) candidate = `${base}:${index++}`;
  usedIds.add(candidate);
  return candidate;
}

function round(value: number): number {
  const result = Math.round(value * 1_000_000_000) / 1_000_000_000;
  return Object.is(result, -0) ? 0 : result;
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function invalid(reason: string, warnings: string[]): BimPastePlan {
  return { valid: false, items: [], selections: [], warnings: unique(warnings), reason };
}
