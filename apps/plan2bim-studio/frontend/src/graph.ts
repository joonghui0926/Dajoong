import type {
  BaseEntity,
  CollectionName,
  CorrectionOperation,
  PlanGraph,
  Selection,
} from "./types";

export const collections: CollectionName[] = [
  "levels",
  "walls",
  "rooms",
  "openings",
  "fixtures",
  "routes",
  "vertical_connections",
  "constraints",
  "dimensions",
];

export function entities(graph: PlanGraph, collection: CollectionName): BaseEntity[] {
  const value = graph[collection];
  return Array.isArray(value) ? (value as BaseEntity[]) : [];
}

export function findEntity(graph: PlanGraph, selection: Selection | null): BaseEntity | null {
  if (!selection) return null;
  return entities(graph, selection.collection).find((item) => item.id === selection.id) ?? null;
}

export function updateEntity(
  graph: PlanGraph,
  selection: Selection,
  changes: Record<string, unknown>,
): PlanGraph {
  const next = structuredClone(graph);
  const items = entities(next, selection.collection);
  const item = items.find((candidate) => candidate.id === selection.id);
  if (!item) return graph;
  Object.assign(item, changes, {
    confidence: 1,
    uncertainty: 0,
    review_state: "accepted",
    model_version: "human-correction",
  });
  if (selection.collection === "walls") {
    const changedHandles = (["from", "to"] as const).filter((handle) =>
      Object.prototype.hasOwnProperty.call(changes, handle),
    );
    if (changedHandles.length) {
      propagateCoincidentConstraints(next, selection.id, changedHandles);
      synchronizeHostedOpenings(next);
    }
  }
  return next;
}

function propagateCoincidentConstraints(
  graph: PlanGraph,
  wallId: string,
  handles: Array<"from" | "to">,
): void {
  const wallMap = new Map(graph.walls.map((wall) => [wall.id, wall]));
  const queue = handles.map((handle) => ({ entity_id: wallId, handle }));
  const visited = new Set<string>();
  while (queue.length) {
    const current = queue.shift();
    if (!current) break;
    const key = `${current.entity_id}:${current.handle}`;
    if (visited.has(key)) continue;
    visited.add(key);
    const sourceWall = wallMap.get(current.entity_id);
    if (!sourceWall) continue;
    const point: [number, number] = [...sourceWall[current.handle]];
    for (const constraint of graph.constraints ?? []) {
      if (constraint.type !== "coincident") continue;
      if (!constraint.references.some(
        (reference) => reference.entity_id === current.entity_id && reference.handle === current.handle,
      )) continue;
      for (const reference of constraint.references) {
        const targetWall = wallMap.get(reference.entity_id);
        if (!targetWall) continue;
        targetWall[reference.handle] = [...point];
        targetWall.confidence = 1;
        targetWall.uncertainty = 0;
        targetWall.review_state = "accepted";
        targetWall.model_version = "human-correction";
        queue.push({ entity_id: reference.entity_id, handle: reference.handle });
      }
    }
  }
}

function synchronizeHostedOpenings(graph: PlanGraph): void {
  const walls = new Map(graph.walls.map((wall) => [wall.id, wall]));
  for (const opening of graph.openings) {
    const wall = walls.get(opening.wall_id);
    if (!wall) continue;
    const dx = wall.to[0] - wall.from[0];
    const dy = wall.to[1] - wall.from[1];
    const length = Math.hypot(dx, dy);
    if (length < 1e-8) continue;
    const halfWidth = Math.min(length / 2, Math.max(0, Number(opening.width_m) / 2));
    const requestedOffset = Number.isFinite(Number(opening.x_m))
      ? Number(opening.x_m)
      : ((opening.center_m[0] - wall.from[0]) * dx + (opening.center_m[1] - wall.from[1]) * dy) / length;
    const offset = Math.max(halfWidth, Math.min(length - halfWidth, requestedOffset));
    opening.x_m = offset;
    opening.center_m = [
      wall.from[0] + dx / length * offset,
      wall.from[1] + dy / length * offset,
    ];
  }
}

export function deleteEntity(graph: PlanGraph, selection: Selection): PlanGraph {
  const next = structuredClone(graph);
  const items = entities(next, selection.collection);
  const index = items.findIndex((candidate) => candidate.id === selection.id);
  if (index >= 0) items.splice(index, 1);
  return next;
}

export function operationFor(
  selection: Selection,
  changes: Record<string, unknown>,
  reason = "manual_review",
): CorrectionOperation {
  return {
    id: `edit-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    action: "update",
    collection: selection.collection,
    entity_id: selection.id,
    changes,
    reason,
  };
}

export function graphBounds(graph: PlanGraph, levelId?: string) {
  const points: [number, number][] = [];
  for (const wall of graph.walls) {
    if (!levelId || wall.level_id === levelId) points.push(wall.from, wall.to);
  }
  for (const room of graph.rooms) {
    if (!levelId || room.level_id === levelId) points.push(...room.polygon);
  }
  for (const route of graph.routes) {
    if (!levelId || route.level_id === levelId) {
      points.push(...route.points_m.map(([x, y]) => [x, y] as [number, number]));
    }
  }
  for (const connection of graph.vertical_connections ?? []) {
    if (
      !levelId ||
      connection.from_level_id === levelId ||
      connection.to_level_id === levelId
    ) {
      const [x, y] = connection.center_m;
      const [width, depth] = connection.footprint_m;
      points.push([x - width / 2, y - depth / 2], [x + width / 2, y + depth / 2]);
    }
  }
  if (!points.length) return { minX: 0, minY: 0, maxX: 10, maxY: 10 };
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  return {
    minX: Math.min(...xs),
    minY: Math.min(...ys),
    maxX: Math.max(...xs),
    maxY: Math.max(...ys),
  };
}

export function confidenceLabel(entity: BaseEntity): string {
  const confidence = typeof entity.confidence === "number" ? entity.confidence : 0;
  return `${Math.round(confidence * 100)}%`;
}

export function downloadJson(name: string, payload: unknown) {
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}
