import { isSimplePolygon, polygonArea, type Point2 } from "./editorGeometry";
import { polygonContainsPolygon, rotatedFixtureFootprint } from "./fixturePlacement";
import type { PlanGraph, RoomEntity, Selection, WallEntity } from "./types";

export interface RoomBoundaryTransformEntry {
  selection: Selection;
  changes: Record<string, unknown>;
}

export interface RoomBoundaryPlan {
  valid: boolean;
  entries: RoomBoundaryTransformEntry[];
  graph?: PlanGraph;
  notices: string[];
  reason?: string;
}

export interface RoomPointSnap {
  point: Point2;
  wallId: string;
  distance: number;
}

const MINIMUM_ROOM_AREA_M2 = 0.05;
const MINIMUM_EDGE_LENGTH_M = 0.02;
const EPSILON = 1e-8;

export function insertRoomBoundaryVertex(
  polygon: Point2[],
  edgeIndex: number,
  point?: Point2,
): { valid: boolean; polygon: Point2[]; vertexIndex?: number; reason?: string } {
  if (polygon.length < 3 || edgeIndex < 0 || edgeIndex >= polygon.length) {
    return { valid: false, polygon, reason: "Select a valid room edge." };
  }
  const start = polygon[edgeIndex];
  const end = polygon[(edgeIndex + 1) % polygon.length];
  const vertex: Point2 = point ?? [
    roundMetric((start[0] + end[0]) / 2),
    roundMetric((start[1] + end[1]) / 2),
  ];
  const vertexIndex = edgeIndex + 1;
  const next = [
    ...polygon.slice(0, vertexIndex),
    vertex,
    ...polygon.slice(vertexIndex),
  ];
  const validation = validateRoomPolygon(next);
  return validation.valid
    ? { valid: true, polygon: next, vertexIndex }
    : { valid: false, polygon, reason: validation.reason };
}

export function removeRoomBoundaryVertex(
  polygon: Point2[],
  vertexIndex: number,
): { valid: boolean; polygon: Point2[]; reason?: string } {
  if (polygon.length <= 3) {
    return { valid: false, polygon, reason: "A room boundary needs at least three vertices." };
  }
  if (vertexIndex < 0 || vertexIndex >= polygon.length) {
    return { valid: false, polygon, reason: "Select a valid room vertex." };
  }
  const next = polygon.filter((_, index) => index !== vertexIndex);
  const validation = validateRoomPolygon(next);
  return validation.valid
    ? { valid: true, polygon: next }
    : { valid: false, polygon, reason: validation.reason };
}

export function snapRoomBoundaryPoint(
  point: Point2,
  walls: WallEntity[],
  toleranceM: number,
): RoomPointSnap | null {
  const nearest = walls
    .map((wall) => {
      const projection = projectPointToSegment(point, wall.from, wall.to);
      return projection ? { ...projection, wallId: wall.id } : null;
    })
    .filter((item): item is RoomPointSnap => Boolean(item))
    .filter((item) => item.distance <= Math.max(0, toleranceM))
    .sort((left, right) => left.distance - right.distance)[0];
  return nearest ?? null;
}

export function planRoomBoundaryTransform(
  graph: PlanGraph,
  roomId: string,
  polygon: Point2[],
): RoomBoundaryPlan {
  const room = graph.rooms.find((item) => item.id === roomId);
  if (!room) return failure(`${roomId} is no longer present in the graph.`);
  const validation = validateRoomPolygon(polygon);
  if (!validation.valid) return failure(validation.reason ?? "Room boundary is invalid.");
  if (samePolygon(room.polygon, polygon)) {
    return { valid: true, entries: [], graph, notices: [] };
  }

  const notices: string[] = [];
  for (const peer of graph.rooms.filter(
    (item) => item.level_id === room.level_id && item.id !== room.id,
  )) {
    const beforeOverlap = polygonsInteriorOverlap(room.polygon, peer.polygon);
    const afterOverlap = polygonsInteriorOverlap(polygon, peer.polygon);
    if (!beforeOverlap && afterOverlap) {
      return failure(`${room.id} would overlap ${peer.id}.`);
    }
    if (beforeOverlap && afterOverlap) {
      notices.push(`${room.id} retains a source overlap with ${peer.id} for review.`);
    }
  }

  const candidate = structuredClone(graph);
  const candidateRoom = candidate.rooms.find((item) => item.id === roomId);
  if (!candidateRoom) return failure(`${roomId} is no longer present in the graph.`);
  candidateRoom.polygon = polygon.map((point) => [...point] as Point2);

  const entries: RoomBoundaryTransformEntry[] = [{
    selection: { collection: "rooms", id: roomId },
    changes: { polygon: candidateRoom.polygon },
  }];
  let reassigned = 0;
  let unassigned = 0;
  const originalRooms = graph.rooms.filter((item) => item.level_id === room.level_id);
  const candidateRooms = candidate.rooms.filter((item) => item.level_id === room.level_id);
  for (const fixture of candidate.fixtures.filter((item) => item.level_id === room.level_id)) {
    const footprint = rotatedFixtureFootprint(
      fixture.center_m,
      fixture.size_m,
      Number(fixture.yaw_deg ?? 0),
    );
    const beforeRoom = smallestContainingRoom(originalRooms, footprint);
    const afterRoom = smallestContainingRoom(candidateRooms, footprint);
    if (beforeRoom?.id === afterRoom?.id) continue;
    const nextRoomId = afterRoom?.id ?? null;
    Object.assign(fixture, { room_id: nextRoomId });
    const changes: Record<string, unknown> = { room_id: nextRoomId };
    if (nextRoomId) {
      reassigned += 1;
    } else {
      unassigned += 1;
      changes.review_state = "review_required";
      changes.uncertainty = Math.max(0.2, Number(fixture.uncertainty ?? 0));
      fixture.review_state = "review_required";
      fixture.uncertainty = changes.uncertainty as number;
    }
    entries.push({ selection: { collection: "fixtures", id: fixture.id }, changes });
  }

  if (reassigned) {
    notices.push(`${reassigned} contained object${reassigned === 1 ? "" : "s"} reassigned.`);
  }
  if (unassigned) {
    notices.push(`${unassigned} object${unassigned === 1 ? "" : "s"} moved to review because no room contains it.`);
  }
  return { valid: true, entries, graph: candidate, notices };
}

export function validateRoomPolygon(
  polygon: Point2[],
): { valid: boolean; reason?: string } {
  if (polygon.length < 3 || polygon.some((point) =>
    point.length < 2 || !Number.isFinite(point[0]) || !Number.isFinite(point[1]),
  )) {
    return { valid: false, reason: "A room boundary needs at least three finite vertices." };
  }
  for (let index = 0; index < polygon.length; index += 1) {
    if (distance(polygon[index], polygon[(index + 1) % polygon.length]) < MINIMUM_EDGE_LENGTH_M) {
      return { valid: false, reason: "Room boundary edges must remain at least 20 mm long." };
    }
  }
  if (!isSimplePolygon(polygon)) {
    return { valid: false, reason: "Room boundary would become self intersecting." };
  }
  if (polygonArea(polygon) < MINIMUM_ROOM_AREA_M2) {
    return { valid: false, reason: "Room area must remain at least 0.05 m²." };
  }
  return { valid: true };
}

function smallestContainingRoom(rooms: RoomEntity[], footprint: Point2[]): RoomEntity | undefined {
  return rooms
    .filter((room) => polygonContainsPolygon(room.polygon, footprint))
    .sort((left, right) => polygonArea(left.polygon) - polygonArea(right.polygon))[0];
}

function polygonsInteriorOverlap(first: Point2[], second: Point2[]): boolean {
  for (let firstIndex = 0; firstIndex < first.length; firstIndex += 1) {
    const firstStart = first[firstIndex];
    const firstEnd = first[(firstIndex + 1) % first.length];
    for (let secondIndex = 0; secondIndex < second.length; secondIndex += 1) {
      const secondStart = second[secondIndex];
      const secondEnd = second[(secondIndex + 1) % second.length];
      if (segmentsProperlyIntersect(firstStart, firstEnd, secondStart, secondEnd)) return true;
    }
  }
  return polygonInteriorSamples(first).some((point) => pointStrictlyInsidePolygon(point, second))
    || polygonInteriorSamples(second).some((point) => pointStrictlyInsidePolygon(point, first));
}

function polygonInteriorSamples(polygon: Point2[]): Point2[] {
  return polygon.flatMap((point, index) => {
    const next = polygon[(index + 1) % polygon.length];
    return [point, [(point[0] + next[0]) / 2, (point[1] + next[1]) / 2] as Point2];
  });
}

function pointStrictlyInsidePolygon(point: Point2, polygon: Point2[]): boolean {
  if (polygon.some((start, index) =>
    pointOnSegment(point, start, polygon[(index + 1) % polygon.length]),
  )) return false;
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const start = polygon[previous];
    const end = polygon[index];
    if ((start[1] > point[1]) !== (end[1] > point[1])
      && point[0] < (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0]) {
      inside = !inside;
    }
  }
  return inside;
}

function segmentsProperlyIntersect(a: Point2, b: Point2, c: Point2, d: Point2): boolean {
  const abC = orientation(a, b, c);
  const abD = orientation(a, b, d);
  const cdA = orientation(c, d, a);
  const cdB = orientation(c, d, b);
  return abC * abD < -EPSILON && cdA * cdB < -EPSILON;
}

function orientation(a: Point2, b: Point2, c: Point2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function pointOnSegment(point: Point2, start: Point2, end: Point2): boolean {
  if (Math.abs(orientation(start, end, point)) > EPSILON) return false;
  return point[0] >= Math.min(start[0], end[0]) - EPSILON
    && point[0] <= Math.max(start[0], end[0]) + EPSILON
    && point[1] >= Math.min(start[1], end[1]) - EPSILON
    && point[1] <= Math.max(start[1], end[1]) + EPSILON;
}

function projectPointToSegment(point: Point2, start: Point2, end: Point2) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < EPSILON) return null;
  const parameter = Math.max(0, Math.min(1,
    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared,
  ));
  const projected: Point2 = [
    roundMetric(start[0] + dx * parameter),
    roundMetric(start[1] + dy * parameter),
  ];
  return { point: projected, distance: distance(point, projected) };
}

function samePolygon(first: Point2[], second: Point2[]): boolean {
  return first.length === second.length
    && first.every((point, index) => distance(point, second[index]) <= EPSILON);
}

function distance(first: Point2, second: Point2): number {
  return Math.hypot(second[0] - first[0], second[1] - first[1]);
}

function roundMetric(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function failure(reason: string): RoomBoundaryPlan {
  return { valid: false, entries: [], notices: [], reason };
}
