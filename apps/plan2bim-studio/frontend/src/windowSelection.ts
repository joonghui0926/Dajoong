import { openingFrame } from "./openingGeometry";
import type { PlanGraph, Selection } from "./types";

export interface SelectionRectangle {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export type WindowSelectionMode = "window" | "crossing";

export interface Footprint {
  points: Array<[number, number]>;
  closed: boolean;
}

export function selectionRectangle(
  start: [number, number],
  current: [number, number],
): SelectionRectangle {
  return {
    minX: Math.min(start[0], current[0]),
    minY: Math.min(start[1], current[1]),
    maxX: Math.max(start[0], current[0]),
    maxY: Math.max(start[1], current[1]),
  };
}

export function selectionMode(
  start: [number, number],
  current: [number, number],
): WindowSelectionMode {
  return current[0] >= start[0] ? "window" : "crossing";
}

export function selectInRectangle(
  graph: PlanGraph,
  candidates: Selection[],
  rectangle: SelectionRectangle,
  mode: WindowSelectionMode,
): Selection[] {
  return candidates.filter((selection) => {
    const footprint = elementFootprint(graph, selection);
    if (!footprint?.points.length) return false;
    return mode === "window"
      ? footprint.points.every((point) => pointInRectangle(point, rectangle))
      : footprintIntersectsRectangle(footprint, rectangle);
  });
}

export function elementFootprint(
  graph: PlanGraph,
  selection: Selection,
): Footprint | null {
  if (selection.collection === "walls") {
    const wall = graph.walls.find((item) => item.id === selection.id);
    if (!wall) return null;
    const dx = wall.to[0] - wall.from[0];
    const dy = wall.to[1] - wall.from[1];
    const length = Math.hypot(dx, dy);
    if (length < 1e-9) return { points: [wall.from], closed: false };
    const half = Math.max(0.03, wall.thickness_m / 2);
    const px = -dy / length * half;
    const py = dx / length * half;
    return {
      points: [
        [wall.from[0] + px, wall.from[1] + py],
        [wall.to[0] + px, wall.to[1] + py],
        [wall.to[0] - px, wall.to[1] - py],
        [wall.from[0] - px, wall.from[1] - py],
      ],
      closed: true,
    };
  }
  if (selection.collection === "rooms") {
    const room = graph.rooms.find((item) => item.id === selection.id);
    return room ? { points: room.polygon, closed: true } : null;
  }
  if (selection.collection === "openings") {
    const opening = graph.openings.find((item) => item.id === selection.id);
    const wall = opening && graph.walls.find((item) => item.id === opening.wall_id);
    const frame = opening && wall ? openingFrame(opening, wall) : null;
    if (!frame) return null;
    const dx = frame.end[0] - frame.start[0];
    const dy = frame.end[1] - frame.start[1];
    const length = Math.max(1e-9, Math.hypot(dx, dy));
    const px = -dy / length * 0.08;
    const py = dx / length * 0.08;
    return {
      points: [
        [frame.start[0] + px, frame.start[1] + py],
        [frame.end[0] + px, frame.end[1] + py],
        [frame.end[0] - px, frame.end[1] - py],
        [frame.start[0] - px, frame.start[1] - py],
      ],
      closed: true,
    };
  }
  if (selection.collection === "fixtures") {
    const fixture = graph.fixtures.find((item) => item.id === selection.id);
    return fixture
      ? { points: rotatedRectangle(fixture.center_m, fixture.size_m[0], fixture.size_m[1], fixture.yaw_deg ?? 0), closed: true }
      : null;
  }
  if (selection.collection === "vertical_connections") {
    const connection = (graph.vertical_connections ?? []).find((item) => item.id === selection.id);
    return connection
      ? { points: rotatedRectangle(connection.center_m, connection.footprint_m[0], connection.footprint_m[1], connection.yaw_deg ?? 0), closed: true }
      : null;
  }
  if (selection.collection === "routes") {
    const route = graph.routes.find((item) => item.id === selection.id);
    return route ? { points: route.points_m.map(([x, y]) => [x, y]), closed: false } : null;
  }
  if (selection.collection === "dimensions") {
    const dimension = (graph.dimensions ?? []).find((item) => item.id === selection.id);
    return dimension ? { points: [dimension.from, dimension.to], closed: false } : null;
  }
  if (selection.collection === "constraints") {
    const constraint = (graph.constraints ?? []).find((item) => item.id === selection.id);
    const reference = constraint?.references[0];
    const wall = reference && graph.walls.find((item) => item.id === reference.entity_id);
    const point = wall && reference ? wall[reference.handle] : null;
    return point ? { points: [point], closed: false } : null;
  }
  return null;
}

function rotatedRectangle(
  center: [number, number],
  width: number,
  depth: number,
  yawDegrees: number,
): Array<[number, number]> {
  const radians = yawDegrees * Math.PI / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  return [
    [-width / 2, -depth / 2],
    [width / 2, -depth / 2],
    [width / 2, depth / 2],
    [-width / 2, depth / 2],
  ].map(([x, y]) => [
    center[0] + x * cosine - y * sine,
    center[1] + x * sine + y * cosine,
  ]);
}

function footprintIntersectsRectangle(
  footprint: Footprint,
  rectangle: SelectionRectangle,
): boolean {
  if (footprint.points.some((point) => pointInRectangle(point, rectangle))) return true;
  const rectangleCorners: Array<[number, number]> = [
    [rectangle.minX, rectangle.minY],
    [rectangle.maxX, rectangle.minY],
    [rectangle.maxX, rectangle.maxY],
    [rectangle.minX, rectangle.maxY],
  ];
  if (footprint.closed && rectangleCorners.some((point) => pointInPolygon(point, footprint.points))) {
    return true;
  }
  const footprintSegments = segments(footprint.points, footprint.closed);
  const rectangleSegments = segments(rectangleCorners, true);
  return footprintSegments.some(([a, b]) =>
    rectangleSegments.some(([c, d]) => segmentsIntersect(a, b, c, d)),
  );
}

function pointInRectangle(point: [number, number], rectangle: SelectionRectangle): boolean {
  return point[0] >= rectangle.minX - 1e-9
    && point[0] <= rectangle.maxX + 1e-9
    && point[1] >= rectangle.minY - 1e-9
    && point[1] <= rectangle.maxY + 1e-9;
}

function segments(
  points: Array<[number, number]>,
  closed: boolean,
): Array<[[number, number], [number, number]]> {
  const result: Array<[[number, number], [number, number]]> = [];
  for (let index = 1; index < points.length; index += 1) {
    result.push([points[index - 1], points[index]]);
  }
  if (closed && points.length > 2) result.push([points.at(-1)!, points[0]]);
  return result;
}

function segmentsIntersect(
  a: [number, number],
  b: [number, number],
  c: [number, number],
  d: [number, number],
): boolean {
  const orientation = (p: [number, number], q: [number, number], r: [number, number]) =>
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const abC = orientation(a, b, c);
  const abD = orientation(a, b, d);
  const cdA = orientation(c, d, a);
  const cdB = orientation(c, d, b);
  if (((abC > 0 && abD < 0) || (abC < 0 && abD > 0))
    && ((cdA > 0 && cdB < 0) || (cdA < 0 && cdB > 0))) return true;
  const onSegment = (p: [number, number], q: [number, number], r: [number, number]) =>
    Math.abs(orientation(p, q, r)) < 1e-9
    && r[0] >= Math.min(p[0], q[0]) - 1e-9
    && r[0] <= Math.max(p[0], q[0]) + 1e-9
    && r[1] >= Math.min(p[1], q[1]) - 1e-9
    && r[1] <= Math.max(p[1], q[1]) + 1e-9;
  return onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b);
}

function pointInPolygon(point: [number, number], polygon: Array<[number, number]>): boolean {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const [xi, yi] = polygon[index];
    const [xj, yj] = polygon[previous];
    const intersects = ((yi > point[1]) !== (yj > point[1]))
      && point[0] < (xj - xi) * (point[1] - yi) / ((yj - yi) || 1e-12) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}
