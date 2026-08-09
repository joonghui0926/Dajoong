import { moveOpeningToPoint, validateOpeningPlacement } from "./openingGeometry";
import type { EntityChanges } from "./editorCommands";
import type {
  CollectionName,
  OpeningEntity,
  PlanGraph,
  Selection,
} from "./types";

export const EXACT_MOVE_COLLECTIONS: CollectionName[] = [
  "walls",
  "openings",
  "fixtures",
  "routes",
  "vertical_connections",
  "dimensions",
];

export interface ExactTranslationResult {
  valid: boolean;
  changesById: EntityChanges;
  notices: string[];
  reason?: string;
}

export function canMoveExactly(selection: Selection): boolean {
  return EXACT_MOVE_COLLECTIONS.includes(selection.collection);
}

export function canRotateExactly(selection: Selection): boolean {
  return selection.collection === "fixtures" || selection.collection === "vertical_connections";
}

export function exactRotationChanges(
  graph: PlanGraph,
  selections: Selection[],
  pivot: [number, number],
  deltaDegrees: number,
): ExactTranslationResult {
  if (!pivot.every(Number.isFinite) || !Number.isFinite(deltaDegrees)) {
    return { valid: false, changesById: {}, notices: [], reason: "Rotation pivot and angle must be finite." };
  }
  if (!selections.length || selections.some((selection) => !canRotateExactly(selection))) {
    return {
      valid: false,
      changesById: {},
      notices: [],
      reason: "Only placed objects and vertical connections support exact group rotation.",
    };
  }
  const radians = deltaDegrees * Math.PI / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const changesById: EntityChanges = {};
  for (const selection of selections) {
    const entity = selection.collection === "fixtures"
      ? graph.fixtures.find((item) => item.id === selection.id)
      : (graph.vertical_connections ?? []).find((item) => item.id === selection.id);
    if (!entity) return missing(selection.id);
    const dx = entity.center_m[0] - pivot[0];
    const dy = entity.center_m[1] - pivot[1];
    const currentYaw = Number.isFinite(Number(entity.yaw_deg)) ? Number(entity.yaw_deg) : 0;
    changesById[entity.id] = {
      center_m: [
        roundMetric(pivot[0] + dx * cosine - dy * sine),
        roundMetric(pivot[1] + dx * sine + dy * cosine),
      ],
      yaw_deg: normalizeDegrees(currentYaw + deltaDegrees),
    };
  }
  return { valid: true, changesById, notices: [] };
}

export function exactTranslationChanges(
  graph: PlanGraph,
  selections: Selection[],
  delta: [number, number],
): ExactTranslationResult {
  if (!delta.every(Number.isFinite)) {
    return { valid: false, changesById: {}, notices: [], reason: "Offsets must be finite." };
  }
  const changesById: EntityChanges = {};
  const notices: string[] = [];
  const selectedWallIds = new Set(
    selections.filter((item) => item.collection === "walls").map((item) => item.id),
  );
  const selectedOpeningIds = new Set(
    selections.filter((item) => item.collection === "openings").map((item) => item.id),
  );

  for (const selection of selections) {
    if (!canMoveExactly(selection)) {
      return {
        valid: false,
        changesById: {},
        notices: [],
        reason: `${selection.collection.replaceAll("_", " ")} cannot be translated as geometry.`,
      };
    }
    if (selection.collection === "walls") {
      const wall = graph.walls.find((item) => item.id === selection.id);
      if (!wall) return missing(selection.id);
      changesById[wall.id] = {
        from: translatePoint(wall.from, delta),
        to: translatePoint(wall.to, delta),
      };
      continue;
    }
    if (selection.collection === "fixtures") {
      const fixture = graph.fixtures.find((item) => item.id === selection.id);
      if (!fixture) return missing(selection.id);
      changesById[fixture.id] = { center_m: translatePoint(fixture.center_m, delta) };
      continue;
    }
    if (selection.collection === "vertical_connections") {
      const connection = (graph.vertical_connections ?? []).find(
        (item) => item.id === selection.id,
      );
      if (!connection) return missing(selection.id);
      changesById[connection.id] = { center_m: translatePoint(connection.center_m, delta) };
      continue;
    }
    if (selection.collection === "routes") {
      const route = graph.routes.find((item) => item.id === selection.id);
      if (!route) return missing(selection.id);
      changesById[route.id] = {
        points_m: route.points_m.map(([x, y, z]) => [
          roundMetric(x + delta[0]),
          roundMetric(y + delta[1]),
          z,
        ]),
      };
      continue;
    }
    if (selection.collection === "dimensions") {
      const dimension = (graph.dimensions ?? []).find((item) => item.id === selection.id);
      if (!dimension) return missing(selection.id);
      changesById[dimension.id] = {
        from: translatePoint(dimension.from, delta),
        to: translatePoint(dimension.to, delta),
      };
    }
  }

  const openingChanges: Record<string, Record<string, unknown>> = {};
  for (const selection of selections.filter((item) => item.collection === "openings")) {
    const opening = graph.openings.find((item) => item.id === selection.id);
    if (!opening) return missing(selection.id);
    if (selectedWallIds.has(opening.wall_id)) {
      notices.push(`${opening.id} follows its translated host wall.`);
      continue;
    }
    const host = graph.walls.find((item) => item.id === opening.wall_id);
    if (!host) {
      return {
        valid: false,
        changesById: {},
        notices: [],
        reason: `${opening.id} has no valid host wall.`,
      };
    }
    const target = translatePoint(opening.center_m, delta);
    const fixedPeers = graph.openings.filter(
      (item) => item.id === opening.id || !selectedOpeningIds.has(item.id),
    );
    const placement = moveOpeningToPoint(opening, host, fixedPeers, target);
    if (!placement.valid || !placement.changes) {
      return {
        valid: false,
        changesById: {},
        notices: [],
        reason: openingMoveReason(opening, placement.reason, placement.conflictId),
      };
    }
    if (Math.hypot(
      placement.changes.center_m[0] - opening.center_m[0],
      placement.changes.center_m[1] - opening.center_m[1],
    ) <= 0.000001) {
      return {
        valid: false,
        changesById: {},
        notices: [],
        reason: `${opening.id} offset has no component along its host wall.`,
      };
    }
    openingChanges[opening.id] = placement.changes;
    if (Math.hypot(
      placement.changes.center_m[0] - target[0],
      placement.changes.center_m[1] - target[1],
    ) > 0.001) {
      notices.push(`${opening.id} movement was projected onto its host wall.`);
    }
  }

  if (Object.keys(openingChanges).length) {
    const finalOpenings: OpeningEntity[] = graph.openings.map((opening) => ({
      ...opening,
      ...(openingChanges[opening.id] ?? {}),
    }));
    for (const openingId of Object.keys(openingChanges)) {
      const opening = finalOpenings.find((item) => item.id === openingId);
      const host = opening && graph.walls.find((item) => item.id === opening.wall_id);
      if (!opening || !host) return missing(openingId);
      const validation = validateOpeningPlacement(opening, host, finalOpenings);
      if (!validation.valid) {
        return {
          valid: false,
          changesById: {},
          notices: [],
          reason: openingMoveReason(opening, validation.reason, validation.conflictId),
        };
      }
      changesById[opening.id] = openingChanges[opening.id];
    }
  }

  return { valid: true, changesById, notices };
}

function translatePoint(
  point: [number, number],
  delta: [number, number],
): [number, number] {
  return [roundMetric(point[0] + delta[0]), roundMetric(point[1] + delta[1])];
}

function openingMoveReason(
  opening: OpeningEntity,
  reason?: string,
  conflictId?: string,
): string {
  if (reason === "overlap") {
    return `${opening.id} would overlap ${conflictId ?? "another opening"}.`;
  }
  if (reason === "outside_wall") return `${opening.id} would leave its host wall.`;
  return `${opening.id} cannot be placed on its host wall.`;
}

function missing(entityId: string): ExactTranslationResult {
  return {
    valid: false,
    changesById: {},
    notices: [],
    reason: `${entityId} is no longer present in the graph.`,
  };
}

function roundMetric(value: number): number {
  const rounded = Math.round(value * 1_000_000_000) / 1_000_000_000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

function normalizeDegrees(value: number): number {
  const normalized = ((value % 360) + 360) % 360;
  return roundMetric(normalized > 180 ? normalized - 360 : normalized);
}
