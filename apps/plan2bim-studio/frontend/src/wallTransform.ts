import { isSimplePolygon, polygonArea, type Point2 } from "./editorGeometry";
import { validateFixtureEntityChanges } from "./fixturePlacement";
import { updateEntity } from "./graph";
import { validateOpeningPlacement } from "./openingGeometry";
import type {
  FixtureEntity,
  PlanGraph,
  Selection,
  WallEntity,
} from "./types";

export interface WallTransformEntry {
  selection: Selection;
  changes: Record<string, unknown>;
}

export interface WallTransformPlan {
  valid: boolean;
  entries: WallTransformEntry[];
  graph?: PlanGraph;
  notices: string[];
  reason?: string;
}

const MINIMUM_WALL_LENGTH_M = 0.05;
const MINIMUM_ROOM_AREA_M2 = 0.05;

export function planWallTransform(
  graph: PlanGraph,
  wallChangesById: Record<string, Record<string, unknown>>,
  additionalChangesById: Record<string, Record<string, unknown>> = {},
): WallTransformPlan {
  const requested = Object.entries(wallChangesById).filter(([, changes]) =>
    Object.prototype.hasOwnProperty.call(changes, "from")
      || Object.prototype.hasOwnProperty.call(changes, "to"),
  );
  if (!requested.length) return { valid: true, entries: [], graph, notices: [] };
  let candidate = structuredClone(graph);
  for (const [wallId, changes] of requested) {
    if (!candidate.walls.some((wall) => wall.id === wallId)) {
      return failure(`${wallId} is no longer present in the graph.`);
    }
    candidate = updateEntity(candidate, { collection: "walls", id: wallId }, changes);
  }

  const changedWallIds = new Set(candidate.walls
    .filter((wall) => {
      const original = graph.walls.find((item) => item.id === wall.id);
      return original && (!samePoint(original.from, wall.from) || !samePoint(original.to, wall.to));
    })
    .map((wall) => wall.id));
  if (!changedWallIds.size) return { valid: true, entries: [], graph, notices: [] };

  for (const wallId of changedWallIds) {
    const wall = candidate.walls.find((item) => item.id === wallId);
    if (!wall || !validPoint(wall.from) || !validPoint(wall.to)) {
      return failure(`${wallId} has invalid endpoint coordinates.`);
    }
    if (distance(wall.from, wall.to) < MINIMUM_WALL_LENGTH_M) {
      return failure(`${wallId} must remain at least 50 mm long.`);
    }
  }

  const roomResult = propagateRoomBoundaries(graph, candidate, changedWallIds);
  if (!roomResult.valid) return failure(roomResult.reason ?? "A room boundary became invalid.");
  candidate.rooms = roomResult.rooms;
  synchronizeWallHostedFixtures(graph, candidate, changedWallIds);

  const explicitOpeningIds = new Set<string>();
  const explicitFixtureIds = new Set<string>();
  for (const [entityId, changes] of Object.entries(additionalChangesById)) {
    const fixture = candidate.fixtures.find((item) => item.id === entityId);
    if (fixture) {
      Object.assign(fixture, changes);
      explicitFixtureIds.add(entityId);
    }
    const opening = candidate.openings.find((item) => item.id === entityId);
    if (opening) {
      Object.assign(opening, changes);
      explicitOpeningIds.add(entityId);
    }
  }

  for (const opening of candidate.openings) {
    if (!changedWallIds.has(opening.wall_id) && !explicitOpeningIds.has(opening.id)) continue;
    const wall = candidate.walls.find((item) => item.id === opening.wall_id);
    if (!wall) return failure(`${opening.id} lost its host wall.`);
    const validation = validateOpeningPlacement(opening, wall, candidate.openings);
    if (!validation.valid) {
      return failure(openingFailure(opening.id, validation.reason, validation.conflictId));
    }
    Object.assign(opening, validation.changes);
  }

  const affectedLevels = new Set(candidate.walls
    .filter((wall) => changedWallIds.has(wall.id))
    .map((wall) => wall.level_id));
  for (const fixture of candidate.fixtures.filter((item) =>
    affectedLevels.has(item.level_id) || explicitFixtureIds.has(item.id),
  )) {
    const baseline = validateFixtureEntityChanges(graph, fixture.id, {});
    const validation = validateFixtureEntityChanges(candidate, fixture.id, {});
    if (!validation.valid) {
      const directlyAffected = Boolean(
        (fixture.host_wall_id && changedWallIds.has(String(fixture.host_wall_id)))
        || (validation.placement?.conflictIds ?? []).some((id) => changedWallIds.has(id)),
      );
      if (baseline.valid || directlyAffected) {
        return failure(validation.reason ?? `${fixture.id} no longer has a valid placement.`);
      }
      continue;
    }
    const changes = validation.changesById[fixture.id];
    if (changes) Object.assign(fixture, changes);
  }

  const entries = diffWallTransform(graph, candidate, wallChangesById, changedWallIds);
  const dependentCount = entries.filter((entry) => entry.selection.collection !== "walls").length;
  return {
    valid: true,
    entries,
    graph: candidate,
    notices: [
      ...(roomResult.notices ?? []),
      ...(dependentCount
        ? [`${dependentCount} hosted or boundary element${dependentCount === 1 ? "" : "s"} updated.`]
        : []),
    ],
  };
}

function propagateRoomBoundaries(
  original: PlanGraph,
  candidate: PlanGraph,
  changedWallIds: Set<string>,
): { valid: boolean; rooms: PlanGraph["rooms"]; notices?: string[]; reason?: string } {
  const endpointMappings: Array<{
    levelId: string;
    from: Point2;
    to: Point2;
    tolerance: number;
  }> = [];
  const translations: Array<{
    levelId: string;
    wall: WallEntity;
    delta: Point2;
    tolerance: number;
  }> = [];
  for (const wallId of changedWallIds) {
    const before = original.walls.find((wall) => wall.id === wallId);
    const after = candidate.walls.find((wall) => wall.id === wallId);
    if (!before || !after) continue;
    const tolerance = Math.max(0.035, before.thickness_m / 2 + 0.015);
    if (!samePoint(before.from, after.from)) {
      endpointMappings.push({ levelId: before.level_id, from: before.from, to: after.from, tolerance });
    }
    if (!samePoint(before.to, after.to)) {
      endpointMappings.push({ levelId: before.level_id, from: before.to, to: after.to, tolerance });
    }
    const fromDelta: Point2 = [after.from[0] - before.from[0], after.from[1] - before.from[1]];
    const toDelta: Point2 = [after.to[0] - before.to[0], after.to[1] - before.to[1]];
    if (distance(fromDelta, toDelta) <= 1e-7 && distance([0, 0], fromDelta) > 1e-7) {
      translations.push({ levelId: before.level_id, wall: before, delta: fromDelta, tolerance });
    }
  }

  const rooms = original.rooms.map((room) => {
    const polygon = room.polygon.map((vertex) => {
      const endpoint = endpointMappings
        .filter((mapping) => mapping.levelId === room.level_id)
        .map((mapping) => ({ mapping, distance: distance(vertex, mapping.from) }))
        .filter(({ mapping, distance: gap }) => gap <= mapping.tolerance)
        .sort((left, right) => left.distance - right.distance)[0]?.mapping;
      if (endpoint) return roundPoint(endpoint.to);
      const translation = translations.find((item) =>
        item.levelId === room.level_id && pointNearSegment(vertex, item.wall.from, item.wall.to, item.tolerance),
      );
      return translation
        ? roundPoint([vertex[0] + translation.delta[0], vertex[1] + translation.delta[1]])
        : [...vertex] as Point2;
    });
    return { ...room, polygon };
  });
  const notices: string[] = [];
  for (const room of rooms) {
    const before = original.rooms.find((item) => item.id === room.id);
    if (!before || samePolygon(before.polygon, room.polygon)) continue;
    const wasSimple = isSimplePolygon(before.polygon);
    const remainsSimple = isSimplePolygon(room.polygon);
    if (wasSimple && !remainsSimple) {
      return { valid: false, rooms: original.rooms, reason: `${room.id} would become self intersecting.` };
    }
    if (!wasSimple && !remainsSimple) {
      notices.push(`${room.id} retains a source boundary anomaly for review.`);
    }
    if (
      polygonArea(before.polygon) >= MINIMUM_ROOM_AREA_M2
      && polygonArea(room.polygon) < MINIMUM_ROOM_AREA_M2
    ) {
      return { valid: false, rooms: original.rooms, reason: `${room.id} would become smaller than 0.05 m².` };
    }
  }
  return { valid: true, rooms, notices };
}

function synchronizeWallHostedFixtures(
  original: PlanGraph,
  candidate: PlanGraph,
  changedWallIds: Set<string>,
): void {
  for (const fixture of candidate.fixtures) {
    const wallId = typeof fixture.host_wall_id === "string" ? fixture.host_wall_id : "";
    if (!wallId || !changedWallIds.has(wallId)) continue;
    const beforeWall = original.walls.find((wall) => wall.id === wallId);
    const afterWall = candidate.walls.find((wall) => wall.id === wallId);
    const beforeFixture = original.fixtures.find((item) => item.id === fixture.id);
    if (!beforeWall || !afterWall || !beforeFixture) continue;
    const beforeBasis = wallBasis(beforeWall);
    const afterBasis = wallBasis(afterWall);
    if (!beforeBasis || !afterBasis) continue;
    const relative: Point2 = [
      beforeFixture.center_m[0] - beforeWall.from[0],
      beforeFixture.center_m[1] - beforeWall.from[1],
    ];
    const parameter = Math.max(0, Math.min(1,
      (relative[0] * beforeBasis.tangent[0] + relative[1] * beforeBasis.tangent[1]) / beforeBasis.length,
    ));
    const sideDistance = relative[0] * beforeBasis.normal[0] + relative[1] * beforeBasis.normal[1];
    fixture.center_m = roundPoint([
      afterWall.from[0] + afterBasis.tangent[0] * afterBasis.length * parameter + afterBasis.normal[0] * sideDistance,
      afterWall.from[1] + afterBasis.tangent[1] * afterBasis.length * parameter + afterBasis.normal[1] * sideDistance,
    ]);
    const angleDelta = afterBasis.angleDeg - beforeBasis.angleDeg;
    fixture.yaw_deg = normalizeDegrees(Number(beforeFixture.yaw_deg ?? 0) + angleDelta);
  }
}

function diffWallTransform(
  original: PlanGraph,
  candidate: PlanGraph,
  requested: Record<string, Record<string, unknown>>,
  changedWallIds: Set<string>,
): WallTransformEntry[] {
  const entries: WallTransformEntry[] = [];
  for (const wall of candidate.walls.filter((item) => changedWallIds.has(item.id))) {
    const before = original.walls.find((item) => item.id === wall.id);
    if (!before) continue;
    const changes: Record<string, unknown> = { ...(requested[wall.id] ?? {}) };
    if (!samePoint(before.from, wall.from)) changes.from = wall.from;
    if (!samePoint(before.to, wall.to)) changes.to = wall.to;
    if (Object.keys(changes).length) entries.push({ selection: { collection: "walls", id: wall.id }, changes });
  }
  for (const opening of candidate.openings) {
    const before = original.openings.find((item) => item.id === opening.id);
    if (!before) continue;
    const changes: Record<string, unknown> = {};
    if (!samePoint(before.center_m, opening.center_m)) changes.center_m = opening.center_m;
    if (Number(before.x_m ?? 0) !== Number(opening.x_m ?? 0)) changes.x_m = opening.x_m;
    if (Object.keys(changes).length) entries.push({ selection: { collection: "openings", id: opening.id }, changes });
  }
  for (const room of candidate.rooms) {
    const before = original.rooms.find((item) => item.id === room.id);
    if (before && !samePolygon(before.polygon, room.polygon)) {
      entries.push({ selection: { collection: "rooms", id: room.id }, changes: { polygon: room.polygon } });
    }
  }
  for (const fixture of candidate.fixtures) {
    const before = original.fixtures.find((item) => item.id === fixture.id);
    if (!before) continue;
    const changes: Record<string, unknown> = {};
    if (!samePoint(before.center_m, fixture.center_m)) changes.center_m = fixture.center_m;
    if (Number(before.yaw_deg ?? 0) !== Number(fixture.yaw_deg ?? 0)) changes.yaw_deg = fixture.yaw_deg;
    for (const key of ["room_id", "host_wall_id", "mounting"] as const) {
      if (before[key] !== fixture[key]) changes[key] = fixture[key];
    }
    if (Object.keys(changes).length) entries.push({ selection: { collection: "fixtures", id: fixture.id }, changes });
  }
  return entries;
}

function openingFailure(openingId: string, reason?: string, conflictId?: string): string {
  if (reason === "overlap") return `${openingId} would overlap ${conflictId ?? "another opening"}.`;
  if (reason === "outside_wall") return `${openingId} would no longer fit inside its host wall.`;
  if (reason === "too_narrow") return `${openingId} has an invalid width.`;
  return `${openingId} lost valid host geometry.`;
}

function failure(reason: string): WallTransformPlan {
  return { valid: false, entries: [], notices: [], reason };
}

function wallBasis(wall: WallEntity) {
  const dx = wall.to[0] - wall.from[0];
  const dy = wall.to[1] - wall.from[1];
  const length = Math.hypot(dx, dy);
  if (length < 1e-9) return null;
  const tangent: Point2 = [dx / length, dy / length];
  return {
    length,
    tangent,
    normal: [-tangent[1], tangent[0]] as Point2,
    angleDeg: Math.atan2(dy, dx) * 180 / Math.PI,
  };
}

function pointNearSegment(point: Point2, start: Point2, end: Point2, tolerance: number): boolean {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-12) return distance(point, start) <= tolerance;
  const parameter = Math.max(0, Math.min(1,
    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared,
  ));
  return distance(point, [start[0] + dx * parameter, start[1] + dy * parameter]) <= tolerance;
}

function validPoint(point: Point2): boolean {
  return Array.isArray(point) && point.length >= 2 && point.every(Number.isFinite);
}

function samePoint(first: Point2, second: Point2): boolean {
  return distance(first, second) <= 1e-8;
}

function samePolygon(first: Point2[], second: Point2[]): boolean {
  return first.length === second.length && first.every((point, index) => samePoint(point, second[index]));
}

function distance(first: Point2, second: Point2): number {
  return Math.hypot(second[0] - first[0], second[1] - first[1]);
}

function roundPoint(point: Point2): Point2 {
  return [roundMetric(point[0]), roundMetric(point[1])];
}

function roundMetric(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function normalizeDegrees(value: number): number {
  const normalized = ((value % 360) + 360) % 360;
  return roundMetric(normalized);
}
