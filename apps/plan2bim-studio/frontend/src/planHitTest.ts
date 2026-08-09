import { findEntity } from "./graph";
import type { PlanGraph, Selection } from "./types";
import { elementFootprint } from "./windowSelection";

const collectionPriority: Record<Selection["collection"], number> = {
  openings: 0,
  fixtures: 1,
  vertical_connections: 2,
  walls: 3,
  routes: 4,
  constraints: 5,
  dimensions: 6,
  rooms: 7,
  levels: 8,
};

export function hitTestPlanGraph(
  graph: PlanGraph,
  candidates: Selection[],
  point: [number, number],
  toleranceM: number,
): Selection[] {
  return candidates
    .filter((selection) => {
      const footprint = elementFootprint(graph, selection);
      if (!footprint?.points.length) return false;
      if (footprint.closed && pointInPolygon(point, footprint.points)) return true;
      return footprintSegments(footprint.points, footprint.closed).some(([start, end]) =>
        distanceToSegment(point, start, end) <= toleranceM,
      ) || (footprint.points.length === 1
        && Math.hypot(point[0] - footprint.points[0][0], point[1] - footprint.points[0][1]) <= toleranceM);
    })
    .sort((left, right) =>
      collectionPriority[left.collection] - collectionPriority[right.collection]
      || left.id.localeCompare(right.id),
    );
}

export function cycleSelectionIndex(current: number, count: number, reverse = false): number {
  if (count <= 0) return 0;
  return (current + (reverse ? -1 : 1) + count) % count;
}

export function selectionCandidateLabel(graph: PlanGraph, selection: Selection): string {
  const entity = findEntity(graph, selection);
  if (!entity) return selection.id;
  const name = entity.name ?? entity.type ?? entity.kind ?? entity.family_id;
  return typeof name === "string" && name.trim()
    ? name.replaceAll("_", " ")
    : selection.collection.replaceAll("_", " ");
}

function footprintSegments(
  points: Array<[number, number]>,
  closed: boolean,
): Array<[[number, number], [number, number]]> {
  const segments: Array<[[number, number], [number, number]]> = [];
  for (let index = 1; index < points.length; index += 1) {
    segments.push([points[index - 1], points[index]]);
  }
  if (closed && points.length > 2) segments.push([points.at(-1)!, points[0]]);
  return segments;
}

function distanceToSegment(
  point: [number, number],
  start: [number, number],
  end: [number, number],
): number {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-12) return Math.hypot(point[0] - start[0], point[1] - start[1]);
  const projection = Math.max(0, Math.min(1,
    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared,
  ));
  return Math.hypot(
    point[0] - (start[0] + projection * dx),
    point[1] - (start[1] + projection * dy),
  );
}

function pointInPolygon(point: [number, number], polygon: Array<[number, number]>): boolean {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [x, y] = polygon[index];
    const [previousX, previousY] = polygon[previous];
    const crosses = ((y > point[1]) !== (previousY > point[1]))
      && point[0] < (previousX - x) * (point[1] - y) / ((previousY - y) || 1e-12) + x;
    if (crosses) inside = !inside;
  }
  return inside;
}
