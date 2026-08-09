import { fixtureFamilies, type FixtureFamily } from "./families";
import type { FixtureEntity, PlanGraph, RoomEntity, WallEntity } from "./types";

export type Point2 = [number, number];

export type FixturePlacementReason =
  | "outside_room"
  | "wall_conflict"
  | "fixture_conflict"
  | "wall_host_required";

export interface FixturePlacement {
  valid: boolean;
  center: Point2;
  yawDeg: number;
  footprint: Point2[];
  roomId?: string;
  hostWallId?: string;
  reason?: FixturePlacementReason;
  conflictIds: string[];
  mounting: FixtureFamily["mounting"];
}

export interface FixturePlacementOptions {
  ignoreFixtureIds?: Iterable<string>;
}

export interface FixtureTransformValidation {
  valid: boolean;
  changesById: Record<string, Record<string, unknown>>;
  notices: string[];
  reason?: string;
  placement?: FixturePlacement;
}

export interface NewFixtureValidation extends FixtureTransformValidation {
  fixtures: FixtureEntity[];
  offset?: Point2;
}

const EPSILON = 1e-7;

export function rotatedFixtureFootprint(
  center: Point2,
  size: [number, number, number] | [number, number],
  yawDeg: number,
): Point2[] {
  const halfWidth = Math.max(0.001, size[0]) / 2;
  const halfDepth = Math.max(0.001, size[1]) / 2;
  const radians = yawDeg * Math.PI / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  return [
    [-halfWidth, -halfDepth],
    [halfWidth, -halfDepth],
    [halfWidth, halfDepth],
    [-halfWidth, halfDepth],
  ].map(([x, y]) => [
    center[0] + x * cosine - y * sine,
    center[1] + x * sine + y * cosine,
  ] as Point2);
}

export function pointInPolygon(point: Point2, polygon: Point2[]): boolean {
  if (polygon.length < 3) return false;
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const a = polygon[previous];
    const b = polygon[index];
    if (pointOnSegment(point, a, b)) return true;
    const intersects = (a[1] > point[1]) !== (b[1] > point[1])
      && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0];
    if (intersects) inside = !inside;
  }
  return inside;
}

export function polygonContainsPolygon(container: Point2[], subject: Point2[]): boolean {
  return subject.every((point) => pointInPolygon(point, container));
}

export function polygonsOverlap(first: Point2[], second: Point2[]): boolean {
  if (first.length < 3 || second.length < 3) return false;
  for (const polygon of [first, second]) {
    for (let index = 0; index < polygon.length; index += 1) {
      const start = polygon[index];
      const end = polygon[(index + 1) % polygon.length];
      const axis: Point2 = [-(end[1] - start[1]), end[0] - start[0]];
      const firstProjection = project(first, axis);
      const secondProjection = project(second, axis);
      if (firstProjection.max <= secondProjection.min + EPSILON
        || secondProjection.max <= firstProjection.min + EPSILON) return false;
    }
  }
  return true;
}

export function fixturePlacementAt(
  graph: PlanGraph,
  family: FixtureFamily,
  levelId: string,
  cursor: Point2,
  requestedYawDeg = 0,
  options: FixturePlacementOptions = {},
): FixturePlacement {
  const rooms = graph.rooms.filter((room) => room.level_id === levelId);
  const walls = graph.walls.filter((wall) => wall.level_id === levelId);
  let center: Point2 = cursor;
  let yawDeg = normalizeAngle(requestedYawDeg);
  let hostWallId: string | undefined;

  if (family.mounting === "wall") {
    const host = nearestWall(cursor, walls);
    const maximumHostDistance = Math.max(0.8, family.size_m[1] + 0.35);
    if (!host || host.distance > maximumHostDistance) {
      return invalidPlacement(family, center, yawDeg, "wall_host_required");
    }
    hostWallId = host.wall.id;
    yawDeg = normalizeAngle(host.angleDeg + (requestedYawDeg >= 180 ? 180 : 0));
    const thickness = Math.max(0.04, host.wall.thickness_m);
    const depth = Math.max(0.01, family.size_m[1]);
    const preferredSide = host.side === 0 ? 1 : host.side;
    const offset = thickness / 2 + depth / 2 + 0.01;
    const sideCandidates = [preferredSide, -preferredSide].map((side) => {
      const candidateCenter: Point2 = [
        host.projected[0] + host.normal[0] * side * offset,
        host.projected[1] + host.normal[1] * side * offset,
      ];
      return {
        center: candidateCenter,
        hasRoom: Boolean(smallestContainingRoom(
          rooms,
          rotatedFixtureFootprint(candidateCenter, family.size_m, yawDeg),
        )),
      };
    });
    center = (sideCandidates.find((candidate) => candidate.hasRoom) ?? sideCandidates[0]).center;
  }

  const footprint = rotatedFixtureFootprint(center, family.size_m, yawDeg);
  const room = smallestContainingRoom(rooms, footprint);
  if (!room) {
    return invalidPlacement(family, center, yawDeg, "outside_room", footprint, hostWallId);
  }

  const wallConflicts = family.mounting === "ceiling"
    ? []
    : walls.filter((wall) => wall.id !== hostWallId && polygonsOverlap(footprint, wallFootprint(wall)));
  if (wallConflicts.length) {
    return invalidPlacement(
      family,
      center,
      yawDeg,
      "wall_conflict",
      footprint,
      hostWallId,
      wallConflicts.map((wall) => wall.id),
      room.id,
    );
  }

  const ignoredFixtureIds = new Set(options.ignoreFixtureIds ?? []);
  const fixtureConflicts = graph.fixtures.filter((fixture) => {
    if (fixture.level_id !== levelId) return false;
    if (ignoredFixtureIds.has(fixture.id)) return false;
    if (!samePlacementBand(family.mounting, fixture)) return false;
    return polygonsOverlap(
      footprint,
      rotatedFixtureFootprint(fixture.center_m, fixture.size_m, fixture.yaw_deg ?? 0),
    );
  });
  if (fixtureConflicts.length) {
    return invalidPlacement(
      family,
      center,
      yawDeg,
      "fixture_conflict",
      footprint,
      hostWallId,
      fixtureConflicts.map((fixture) => fixture.id),
      room.id,
    );
  }

  return {
    valid: true,
    center,
    yawDeg,
    footprint,
    roomId: room.id,
    hostWallId,
    conflictIds: [],
    mounting: family.mounting,
  };
}

export function fixtureFamilyForEntity(fixture: FixtureEntity): FixtureFamily {
  const registered = fixtureFamilies.find((family) => family.id === fixture.family_id);
  const mounting = registered?.mounting
    ?? (fixture.mounting === "wall" || fixture.mounting === "ceiling" || fixture.mounting === "floor"
      ? fixture.mounting
      : fixture.host_wall_id
        ? "wall"
        : (fixture.base_elevation_m ?? 0) > 1.8
          ? "ceiling"
          : "floor");
  const discipline = registered?.discipline
    ?? (fixture.discipline === "mechanical"
      || fixture.discipline === "electrical"
      || fixture.discipline === "plumbing"
      || fixture.discipline === "fire"
      || fixture.discipline === "architectural"
      ? fixture.discipline
      : "architectural");
  return {
    id: String(fixture.family_id ?? registered?.id ?? `legacy:${fixture.type}`),
    name: registered?.name ?? String(fixture.type || "Component"),
    category: registered?.category ?? "Imported components",
    type: fixture.type,
    discipline,
    size_m: fixture.size_m,
    material: String(fixture.material ?? registered?.material ?? "unspecified"),
    keywords: registered?.keywords ?? [fixture.type],
    mounting,
  };
}

export function validateFixtureTransformChanges(
  graph: PlanGraph,
  changesById: Record<string, Record<string, unknown>>,
): FixtureTransformValidation {
  const fixtureChanges = Object.entries(changesById).filter(([id]) =>
    graph.fixtures.some((fixture) => fixture.id === id),
  );
  if (!fixtureChanges.length) return { valid: true, changesById, notices: [] };
  const changedIds = new Set(fixtureChanges.map(([id]) => id));
  const candidateFixtures = graph.fixtures.map((fixture) => {
    const changes = changesById[fixture.id];
    return changes ? ({ ...fixture, ...changes } as FixtureEntity) : fixture;
  });
  const candidateGraph: PlanGraph = { ...graph, fixtures: candidateFixtures };
  const normalizedById: Record<string, Record<string, unknown>> = { ...changesById };
  const notices: string[] = [];

  for (const [id, originalChanges] of fixtureChanges) {
    const fixture = candidateGraph.fixtures.find((item) => item.id === id);
    if (!fixture) continue;
    if (!validFixtureGeometry(fixture)) {
      return {
        valid: false,
        changesById: {},
        notices: [],
        reason: `${id} has invalid component dimensions or coordinates.`,
      };
    }
    const placement = fixturePlacementAt(
      candidateGraph,
      fixtureFamilyForEntity(fixture),
      fixture.level_id,
      fixture.center_m,
      fixture.yaw_deg ?? 0,
      { ignoreFixtureIds: changedIds },
    );
    if (!placement.valid) return fixtureTransformFailure(id, placement);
    normalizedById[id] = normalizedFixtureChanges(originalChanges, placement);
    const index = candidateGraph.fixtures.findIndex((item) => item.id === id);
    candidateGraph.fixtures[index] = {
      ...fixture,
      ...normalizedById[id],
    } as FixtureEntity;
    if (distance(fixture.center_m, placement.center) > 0.001) {
      notices.push(`${id} snapped to its host.`);
    }
    if (Math.abs(normalizeAngle(Number(fixture.yaw_deg ?? 0)) - placement.yawDeg) > 0.001) {
      notices.push(`${id} rotation aligned to its host.`);
    }
  }

  for (const [id] of fixtureChanges) {
    const fixture = candidateGraph.fixtures.find((item) => item.id === id);
    if (!fixture) continue;
    const placement = fixturePlacementAt(
      candidateGraph,
      fixtureFamilyForEntity(fixture),
      fixture.level_id,
      fixture.center_m,
      fixture.yaw_deg ?? 0,
      { ignoreFixtureIds: [id] },
    );
    if (!placement.valid) return fixtureTransformFailure(id, placement);
    normalizedById[id] = normalizedFixtureChanges(normalizedById[id], placement);
  }
  return { valid: true, changesById: normalizedById, notices };
}

export function validateFixtureEntityChanges(
  graph: PlanGraph,
  fixtureId: string,
  changes: Record<string, unknown>,
): FixtureTransformValidation {
  const result = validateFixtureTransformChanges(graph, { [fixtureId]: changes });
  return {
    ...result,
    placement: result.valid
      ? fixturePlacementForValidatedChanges(graph, fixtureId, result.changesById[fixtureId] ?? {})
      : result.placement,
  };
}

export function validateNewFixtures(
  graph: PlanGraph,
  fixtures: FixtureEntity[],
): NewFixtureValidation {
  if (!fixtures.length) return { valid: true, changesById: {}, notices: [], fixtures: [] };
  const ids = new Set(fixtures.map((fixture) => fixture.id));
  if (ids.size !== fixtures.length || graph.fixtures.some((fixture) => ids.has(fixture.id))) {
    return {
      valid: false,
      changesById: {},
      notices: [],
      fixtures: [],
      reason: "New components require unique IDs.",
    };
  }
  const augmented: PlanGraph = { ...graph, fixtures: [...graph.fixtures, ...fixtures] };
  const changes = Object.fromEntries(fixtures.map((fixture) => [fixture.id, {}]));
  const validation = validateFixtureTransformChanges(augmented, changes);
  if (!validation.valid) return { ...validation, fixtures: [] };
  return {
    ...validation,
    fixtures: fixtures.map((fixture) => ({
      ...fixture,
      ...(validation.changesById[fixture.id] ?? {}),
    } as FixtureEntity)),
  };
}

export function findNearestValidFixtureCopy(
  graph: PlanGraph,
  fixtures: FixtureEntity[],
  snapIncrementM: number,
): NewFixtureValidation {
  if (!fixtures.length) return { valid: true, changesById: {}, notices: [], fixtures: [], offset: [0, 0] };
  const spacing = Math.max(
    Math.max(0.01, snapIncrementM),
    ...fixtures.map((fixture) => Math.max(fixture.size_m[0], fixture.size_m[1]) + Math.max(0.01, snapIncrementM)),
  );
  const directions: Point2[] = [
    [1, 0], [1, 1], [0, 1], [-1, 1],
    [-1, 0], [-1, -1], [0, -1], [1, -1],
  ];
  let lastFailure: NewFixtureValidation | null = null;
  for (let ring = 1; ring <= 8; ring += 1) {
    for (const direction of directions) {
      const offset: Point2 = [direction[0] * spacing * ring, direction[1] * spacing * ring];
      const candidates = fixtures.map((fixture) => ({
        ...fixture,
        center_m: [fixture.center_m[0] + offset[0], fixture.center_m[1] + offset[1]],
      } as FixtureEntity));
      const validation = validateNewFixtures(graph, candidates);
      if (validation.valid) return { ...validation, offset };
      lastFailure = validation;
    }
  }
  return lastFailure ?? {
    valid: false,
    changesById: {},
    notices: [],
    fixtures: [],
    reason: "No clear duplicate placement was found on this level.",
  };
}

export function fixturePlacementMessage(placement: FixturePlacement): string {
  if (placement.valid) return placement.roomId
    ? `Ready in ${placement.roomId}`
    : "Ready to place";
  if (placement.reason === "wall_host_required") return "Move within 800 mm of a host wall";
  if (placement.reason === "outside_room") return "Component must fit fully inside a room";
  if (placement.reason === "wall_conflict") return `Wall collision${placement.conflictIds[0] ? ` · ${placement.conflictIds[0]}` : ""}`;
  return `Object collision${placement.conflictIds[0] ? ` · ${placement.conflictIds[0]}` : ""}`;
}

function invalidPlacement(
  family: FixtureFamily,
  center: Point2,
  yawDeg: number,
  reason: FixturePlacementReason,
  footprint = rotatedFixtureFootprint(center, family.size_m, yawDeg),
  hostWallId?: string,
  conflictIds: string[] = [],
  roomId?: string,
): FixturePlacement {
  return {
    valid: false,
    center,
    yawDeg,
    footprint,
    roomId,
    hostWallId,
    reason,
    conflictIds,
    mounting: family.mounting,
  };
}

function fixturePlacementForValidatedChanges(
  graph: PlanGraph,
  fixtureId: string,
  changes: Record<string, unknown>,
): FixturePlacement | undefined {
  const fixture = graph.fixtures.find((item) => item.id === fixtureId);
  if (!fixture) return undefined;
  const candidate = { ...fixture, ...changes } as FixtureEntity;
  return fixturePlacementAt(
    { ...graph, fixtures: graph.fixtures.map((item) => item.id === fixtureId ? candidate : item) },
    fixtureFamilyForEntity(candidate),
    candidate.level_id,
    candidate.center_m,
    candidate.yaw_deg ?? 0,
    { ignoreFixtureIds: [fixtureId] },
  );
}

function normalizedFixtureChanges(
  changes: Record<string, unknown>,
  placement: FixturePlacement,
): Record<string, unknown> {
  return {
    ...changes,
    center_m: placement.center,
    yaw_deg: placement.yawDeg,
    room_id: placement.roomId,
    host_wall_id: placement.hostWallId,
    mounting: placement.mounting,
  };
}

function fixtureTransformFailure(
  fixtureId: string,
  placement: FixturePlacement,
): FixtureTransformValidation {
  return {
    valid: false,
    changesById: {},
    notices: [],
    reason: `${fixtureId}: ${fixturePlacementMessage(placement)}`,
    placement,
  };
}

function validFixtureGeometry(fixture: FixtureEntity): boolean {
  return fixture.center_m.length >= 2
    && fixture.center_m.every(Number.isFinite)
    && fixture.size_m.length >= 3
    && fixture.size_m.every((value) => Number.isFinite(value) && value > 0)
    && Number.isFinite(Number(fixture.yaw_deg ?? 0));
}

function distance(first: Point2, second: Point2): number {
  return Math.hypot(second[0] - first[0], second[1] - first[1]);
}

function smallestContainingRoom(rooms: RoomEntity[], footprint: Point2[]): RoomEntity | undefined {
  return rooms
    .filter((room) => polygonContainsPolygon(room.polygon, footprint))
    .sort((first, second) => polygonArea(first.polygon) - polygonArea(second.polygon))[0];
}

function wallFootprint(wall: WallEntity): Point2[] {
  const dx = wall.to[0] - wall.from[0];
  const dy = wall.to[1] - wall.from[1];
  const length = Math.max(EPSILON, Math.hypot(dx, dy));
  const normal: Point2 = [-dy / length, dx / length];
  const half = Math.max(0.04, wall.thickness_m) / 2;
  return [
    [wall.from[0] + normal[0] * half, wall.from[1] + normal[1] * half],
    [wall.to[0] + normal[0] * half, wall.to[1] + normal[1] * half],
    [wall.to[0] - normal[0] * half, wall.to[1] - normal[1] * half],
    [wall.from[0] - normal[0] * half, wall.from[1] - normal[1] * half],
  ];
}

function nearestWall(cursor: Point2, walls: WallEntity[]) {
  return walls.map((wall) => {
    const dx = wall.to[0] - wall.from[0];
    const dy = wall.to[1] - wall.from[1];
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared <= EPSILON) return null;
    const ratio = Math.max(0, Math.min(1,
      ((cursor[0] - wall.from[0]) * dx + (cursor[1] - wall.from[1]) * dy) / lengthSquared,
    ));
    const projected: Point2 = [wall.from[0] + dx * ratio, wall.from[1] + dy * ratio];
    const length = Math.sqrt(lengthSquared);
    const normal: Point2 = [-dy / length, dx / length];
    const offset: Point2 = [cursor[0] - projected[0], cursor[1] - projected[1]];
    const signedDistance = offset[0] * normal[0] + offset[1] * normal[1];
    return {
      wall,
      projected,
      normal,
      side: Math.sign(signedDistance),
      distance: Math.hypot(offset[0], offset[1]),
      angleDeg: Math.atan2(dy, dx) * 180 / Math.PI,
    };
  }).filter((value): value is NonNullable<typeof value> => value !== null)
    .sort((first, second) => first.distance - second.distance)[0];
}

function samePlacementBand(mounting: FixtureFamily["mounting"], fixture: FixtureEntity): boolean {
  const existing = fixtureFamilies.find((family) => family.id === fixture.family_id)?.mounting
    ?? ((fixture.base_elevation_m ?? 0) > 1.8 ? "ceiling" : "floor");
  return mounting === "ceiling" ? existing === "ceiling" : existing !== "ceiling";
}

function pointOnSegment(point: Point2, start: Point2, end: Point2): boolean {
  const cross = (point[1] - start[1]) * (end[0] - start[0])
    - (point[0] - start[0]) * (end[1] - start[1]);
  if (Math.abs(cross) > EPSILON) return false;
  const dot = (point[0] - start[0]) * (end[0] - start[0])
    + (point[1] - start[1]) * (end[1] - start[1]);
  if (dot < -EPSILON) return false;
  const lengthSquared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2;
  return dot <= lengthSquared + EPSILON;
}

function project(polygon: Point2[], axis: Point2) {
  const values = polygon.map((point) => point[0] * axis[0] + point[1] * axis[1]);
  return { min: Math.min(...values), max: Math.max(...values) };
}

function polygonArea(polygon: Point2[]): number {
  return Math.abs(polygon.reduce((area, point, index) => {
    const next = polygon[(index + 1) % polygon.length];
    return area + point[0] * next[1] - next[0] * point[1];
  }, 0) / 2);
}

function normalizeAngle(value: number): number {
  return ((value % 360) + 360) % 360;
}
