import type { OpeningEntity, WallEntity } from "./types";

export type Point2 = [number, number];
export type DoorHanding = "start" | "end";
export type DoorSwingSide = "positive" | "negative";

export interface OpeningFrame {
  center: Point2;
  start: Point2;
  end: Point2;
  hinge: Point2;
  latch: Point2;
  openLeafEnd: Point2;
  tangent: Point2;
  normal: Point2;
  width: number;
  wallOffset: number;
  arcPath: string;
}

export interface OpeningResizeResult {
  valid: boolean;
  changes?: {
    center_m: Point2;
    width_m: number;
    x_m: number;
  };
  reason?: "too_narrow" | "invalid_wall";
}

export type OpeningPlacementReason =
  | "invalid_wall"
  | "too_narrow"
  | "outside_wall"
  | "overlap";

export interface OpeningPlacementResult {
  valid: boolean;
  changes?: {
    center_m: Point2;
    width_m: number;
    x_m: number;
  };
  reason?: OpeningPlacementReason;
  conflictId?: string;
}

interface WallBasis {
  length: number;
  tangent: Point2;
}

function wallBasis(wall: WallEntity): WallBasis | null {
  const dx = wall.to[0] - wall.from[0];
  const dy = wall.to[1] - wall.from[1];
  const length = Math.hypot(dx, dy);
  if (length < 1e-8) return null;
  return { length, tangent: [dx / length, dy / length] };
}

function rawWallOffset(point: Point2, wall: WallEntity, basis: WallBasis): number {
  return (
    (point[0] - wall.from[0]) * basis.tangent[0] +
    (point[1] - wall.from[1]) * basis.tangent[1]
  );
}

function pointAtOffset(wall: WallEntity, basis: WallBasis, offset: number): Point2 {
  return [
    wall.from[0] + basis.tangent[0] * offset,
    wall.from[1] + basis.tangent[1] * offset,
  ];
}

export function openingFrame(opening: OpeningEntity, wall: WallEntity): OpeningFrame | null {
  const dx = wall.to[0] - wall.from[0];
  const dy = wall.to[1] - wall.from[1];
  const wallLength = Math.hypot(dx, dy);
  if (wallLength < 1e-8) return null;
  const tangent: Point2 = [dx / wallLength, dy / wallLength];
  const normal: Point2 = [-tangent[1], tangent[0]];
  const projected = projectPointToWall(opening.center_m, wall);
  if (!projected) return null;
  const width = Math.max(0.01, Math.min(Number(opening.width_m), wallLength));
  const half = width / 2;
  const start: Point2 = [
    projected.point[0] - tangent[0] * half,
    projected.point[1] - tangent[1] * half,
  ];
  const end: Point2 = [
    projected.point[0] + tangent[0] * half,
    projected.point[1] + tangent[1] * half,
  ];
  const handing: DoorHanding = opening.handing === "end" ? "end" : "start";
  const swingSide: DoorSwingSide = opening.swing_side === "negative" ? "negative" : "positive";
  const hinge = handing === "start" ? start : end;
  const latch = handing === "start" ? end : start;
  const side = swingSide === "positive" ? 1 : -1;
  const openLeafEnd: Point2 = [
    hinge[0] + normal[0] * width * side,
    hinge[1] + normal[1] * width * side,
  ];
  const closedVector: Point2 = [latch[0] - hinge[0], latch[1] - hinge[1]];
  const openVector: Point2 = [openLeafEnd[0] - hinge[0], openLeafEnd[1] - hinge[1]];
  const cross = closedVector[0] * openVector[1] - closedVector[1] * openVector[0];
  const sweep = cross >= 0 ? 1 : 0;
  const arcPath = `M ${latch[0]} ${latch[1]} A ${width} ${width} 0 0 ${sweep} ${openLeafEnd[0]} ${openLeafEnd[1]}`;
  return {
    center: projected.point,
    start,
    end,
    hinge,
    latch,
    openLeafEnd,
    tangent,
    normal,
    width,
    wallOffset: projected.offset,
    arcPath,
  };
}

export function projectPointToWall(point: Point2, wall: WallEntity): { point: Point2; offset: number; parameter: number } | null {
  const dx = wall.to[0] - wall.from[0];
  const dy = wall.to[1] - wall.from[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-12) return null;
  const parameter = Math.max(
    0,
    Math.min(1, ((point[0] - wall.from[0]) * dx + (point[1] - wall.from[1]) * dy) / lengthSquared),
  );
  const length = Math.sqrt(lengthSquared);
  return {
    point: [wall.from[0] + dx * parameter, wall.from[1] + dy * parameter],
    offset: length * parameter,
    parameter,
  };
}

export function resizeOpeningFromEdge(
  movingPoint: Point2,
  fixedPoint: Point2,
  wall: WallEntity,
  minimumWidth = 0.2,
): OpeningResizeResult {
  const moving = projectPointToWall(movingPoint, wall);
  const fixed = projectPointToWall(fixedPoint, wall);
  if (!moving || !fixed) return { valid: false, reason: "invalid_wall" };
  const width = Math.abs(moving.offset - fixed.offset);
  if (width < minimumWidth) return { valid: false, reason: "too_narrow" };
  const offset = (moving.offset + fixed.offset) / 2;
  const center: Point2 = [
    (moving.point[0] + fixed.point[0]) / 2,
    (moving.point[1] + fixed.point[1]) / 2,
  ];
  return {
    valid: true,
    changes: {
      center_m: center,
      width_m: width,
      x_m: offset,
    },
  };
}

export function validateOpeningPlacement(
  opening: OpeningEntity,
  wall: WallEntity,
  siblings: OpeningEntity[],
  minimumWidth = 0.2,
  minimumGap = 0.02,
): OpeningPlacementResult {
  const basis = wallBasis(wall);
  if (!basis) return { valid: false, reason: "invalid_wall" };
  const width = Number(opening.width_m);
  if (!Number.isFinite(width) || width < minimumWidth) {
    return { valid: false, reason: "too_narrow" };
  }
  const center = opening.center_m;
  if (!Array.isArray(center) || !center.every(Number.isFinite)) {
    return { valid: false, reason: "invalid_wall" };
  }
  const offset = rawWallOffset(center, wall, basis);
  const start = offset - width / 2;
  const end = offset + width / 2;
  if (start < -1e-6 || end > basis.length + 1e-6) {
    return { valid: false, reason: "outside_wall" };
  }
  for (const sibling of siblings) {
    if (sibling.id === opening.id || sibling.wall_id !== wall.id) continue;
    const siblingWidth = Number(sibling.width_m);
    if (!Number.isFinite(siblingWidth) || siblingWidth <= 0) continue;
    const siblingOffset = rawWallOffset(sibling.center_m, wall, basis);
    const siblingStart = siblingOffset - siblingWidth / 2;
    const siblingEnd = siblingOffset + siblingWidth / 2;
    if (start < siblingEnd + minimumGap && end > siblingStart - minimumGap) {
      return { valid: false, reason: "overlap", conflictId: sibling.id };
    }
  }
  return {
    valid: true,
    changes: {
      center_m: pointAtOffset(wall, basis, offset),
      width_m: width,
      x_m: offset,
    },
  };
}

export function moveOpeningToPoint(
  opening: OpeningEntity,
  wall: WallEntity,
  siblings: OpeningEntity[],
  point: Point2,
): OpeningPlacementResult {
  const basis = wallBasis(wall);
  if (!basis) return { valid: false, reason: "invalid_wall" };
  const width = Number(opening.width_m);
  if (!Number.isFinite(width) || width < 0.2 || width > basis.length) {
    return { valid: false, reason: width < 0.2 ? "too_narrow" : "outside_wall" };
  }
  const half = width / 2;
  const offset = Math.max(half, Math.min(basis.length - half, rawWallOffset(point, wall, basis)));
  return validateOpeningPlacement(
    { ...opening, center_m: pointAtOffset(wall, basis, offset), x_m: offset },
    wall,
    siblings,
  );
}

export function findAvailableOpeningPlacement(
  wall: WallEntity,
  siblings: OpeningEntity[],
  width: number,
  minimumGap = 0.05,
): OpeningPlacementResult {
  const basis = wallBasis(wall);
  if (!basis) return { valid: false, reason: "invalid_wall" };
  if (!Number.isFinite(width) || width < 0.2) return { valid: false, reason: "too_narrow" };
  const occupied = siblings
    .filter((item) => item.wall_id === wall.id && Number(item.width_m) > 0)
    .map((item) => {
      const offset = rawWallOffset(item.center_m, wall, basis);
      return [
        Math.max(0, offset - Number(item.width_m) / 2 - minimumGap),
        Math.min(basis.length, offset + Number(item.width_m) / 2 + minimumGap),
      ] as const;
    })
    .sort((a, b) => a[0] - b[0]);
  const gaps: Array<[number, number]> = [];
  let cursor = minimumGap;
  for (const [start, end] of occupied) {
    if (start > cursor) gaps.push([cursor, start]);
    cursor = Math.max(cursor, end);
  }
  if (cursor < basis.length - minimumGap) gaps.push([cursor, basis.length - minimumGap]);
  const gap = gaps
    .filter(([start, end]) => end - start >= width)
    .sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]))[0];
  if (!gap) return { valid: false, reason: "overlap" };
  const offset = (gap[0] + gap[1]) / 2;
  const candidate: OpeningEntity = {
    id: "opening-placement-candidate",
    level_id: wall.level_id,
    type: "door",
    wall_id: wall.id,
    center_m: pointAtOffset(wall, basis, offset),
    width_m: width,
    height_m: 2.1,
  };
  return validateOpeningPlacement(candidate, wall, siblings, 0.2, minimumGap);
}

export function findNearestOpeningPlacement(
  opening: OpeningEntity,
  wall: WallEntity,
  siblings: OpeningEntity[],
  desiredPoint: Point2,
  minimumGap = 0.05,
): OpeningPlacementResult {
  const basis = wallBasis(wall);
  if (!basis) return { valid: false, reason: "invalid_wall" };
  const width = Number(opening.width_m);
  if (!Number.isFinite(width) || width < 0.2) return { valid: false, reason: "too_narrow" };
  if (width + minimumGap * 2 > basis.length) return { valid: false, reason: "outside_wall" };
  const desiredOffset = Math.max(
    width / 2 + minimumGap,
    Math.min(basis.length - width / 2 - minimumGap, rawWallOffset(desiredPoint, wall, basis)),
  );
  const occupied = siblings
    .filter((item) => item.id !== opening.id && item.wall_id === wall.id && Number(item.width_m) > 0)
    .map((item) => {
      const offset = rawWallOffset(item.center_m, wall, basis);
      return [
        Math.max(0, offset - Number(item.width_m) / 2 - minimumGap),
        Math.min(basis.length, offset + Number(item.width_m) / 2 + minimumGap),
      ] as const;
    })
    .sort((left, right) => left[0] - right[0]);
  const gaps: Array<[number, number]> = [];
  let cursor = minimumGap;
  for (const [start, end] of occupied) {
    if (start > cursor) gaps.push([cursor, start]);
    cursor = Math.max(cursor, end);
  }
  if (cursor < basis.length - minimumGap) gaps.push([cursor, basis.length - minimumGap]);
  const candidate = gaps
    .filter(([start, end]) => end - start >= width)
    .map(([start, end]) => {
      const offset = Math.max(start + width / 2, Math.min(end - width / 2, desiredOffset));
      return { offset, distance: Math.abs(offset - desiredOffset) };
    })
    .sort((left, right) => left.distance - right.distance)[0];
  if (!candidate) return { valid: false, reason: "overlap" };
  return validateOpeningPlacement(
    {
      ...opening,
      wall_id: wall.id,
      level_id: wall.level_id,
      center_m: pointAtOffset(wall, basis, candidate.offset),
      x_m: candidate.offset,
    },
    wall,
    siblings,
    0.2,
    minimumGap,
  );
}

export function toggleDoorHanding(value: unknown): DoorHanding {
  if (value === "unknown" || value == null) return "start";
  return value === "end" ? "start" : "end";
}

export function toggleDoorSwingSide(value: unknown): DoorSwingSide {
  if (value === "unknown" || value == null) return "positive";
  return value === "negative" ? "positive" : "negative";
}
