export const MAX_VISIBLE_ROOM_HANDLES = 16;

type Point2 = [number, number];

function pointSegmentDistance(point: Point2, start: Point2, end: Point2): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared <= Number.EPSILON) return Math.hypot(point[0] - start[0], point[1] - start[1]);
  const t = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared));
  return Math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy));
}

function simplifyOpenIndices(points: Point2[], indices: number[], tolerance: number): number[] {
  if (indices.length <= 2) return indices;
  const startIndex = indices[0];
  const endIndex = indices.at(-1)!;
  let furthest = -1;
  let furthestDistance = tolerance;
  for (let cursor = 1; cursor < indices.length - 1; cursor += 1) {
    const distance = pointSegmentDistance(points[indices[cursor]], points[startIndex], points[endIndex]);
    if (distance > furthestDistance) {
      furthest = cursor;
      furthestDistance = distance;
    }
  }
  if (furthest < 0) return [startIndex, endIndex];
  const left = simplifyOpenIndices(points, indices.slice(0, furthest + 1), tolerance);
  const right = simplifyOpenIndices(points, indices.slice(furthest), tolerance);
  return [...left.slice(0, -1), ...right];
}

function simplifiedClosedIndices(points: Point2[], tolerance: number): number[] {
  if (points.length <= 3) return points.map((_, index) => index);
  let splitIndex = 1;
  let splitDistance = -1;
  for (let index = 1; index < points.length; index += 1) {
    const distance = Math.hypot(points[index][0] - points[0][0], points[index][1] - points[0][1]);
    if (distance > splitDistance) {
      splitIndex = index;
      splitDistance = distance;
    }
  }
  const forward = Array.from({ length: splitIndex + 1 }, (_, index) => index);
  const backward = [0, ...Array.from({ length: points.length - splitIndex }, (_, offset) => points.length - 1 - offset)];
  return [...new Set([
    ...simplifyOpenIndices(points, forward, tolerance),
    ...simplifyOpenIndices(points, backward, tolerance),
  ])].sort((a, b) => a - b);
}

export function visibleRoomHandleIndices(
  polygon: Point2[],
  maxHandles = MAX_VISIBLE_ROOM_HANDLES,
): number[] {
  if (polygon.length <= maxHandles) return polygon.map((_, index) => index);
  const xs = polygon.map((point) => point[0]);
  const ys = polygon.map((point) => point[1]);
  const diagonal = Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  let low = 0;
  let high = Math.max(diagonal, 0.001);
  let best = simplifiedClosedIndices(polygon, high);
  for (let iteration = 0; iteration < 24; iteration += 1) {
    const tolerance = (low + high) / 2;
    const candidate = simplifiedClosedIndices(polygon, tolerance);
    if (candidate.length > maxHandles) low = tolerance;
    else {
      best = candidate;
      high = tolerance;
    }
  }
  return best.length >= 3 ? best : [0, Math.floor(polygon.length / 3), Math.floor(2 * polygon.length / 3)];
}
