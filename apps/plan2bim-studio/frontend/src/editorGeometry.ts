import type { PlanGraph, WallEntity } from "./types";

export type Point2 = [number, number];

export interface SmartGuide {
  axis: "x" | "y";
  value: number;
  label: string;
}

export interface SnapOptions {
  grid_m: number;
  tolerance_m: number;
  origin?: Point2;
  orthogonal?: boolean;
  disabled?: boolean;
}

export interface SnapResult {
  point: Point2;
  guides: SmartGuide[];
  label: string;
}

interface Candidate {
  point: Point2;
  label: string;
}

export function smartSnap(
  rawPoint: Point2,
  graph: PlanGraph,
  levelId: string,
  options: SnapOptions,
): SnapResult {
  if (options.disabled) return { point: rawPoint, guides: [], label: "Free" };
  let point: Point2 = options.grid_m > 0
    ? [
        roundMetric(Math.round(rawPoint[0] / options.grid_m) * options.grid_m),
        roundMetric(Math.round(rawPoint[1] / options.grid_m) * options.grid_m),
      ]
    : [...rawPoint];
  const guides: SmartGuide[] = [];
  let label = options.grid_m > 0 ? `${Math.round(options.grid_m * 1000)} mm grid` : "Free";

  if (options.origin && options.orthogonal) {
    const dx = Math.abs(point[0] - options.origin[0]);
    const dy = Math.abs(point[1] - options.origin[1]);
    if (dx >= dy) {
      point = [point[0], options.origin[1]];
      guides.push({ axis: "y", value: options.origin[1], label: "Horizontal" });
    } else {
      point = [options.origin[0], point[1]];
      guides.push({ axis: "x", value: options.origin[0], label: "Vertical" });
    }
    label = "Orthogonal";
  }

  const candidates = snapCandidates(graph, levelId);
  const nearest = candidates
    .map((candidate) => ({
      ...candidate,
      distance: Math.hypot(candidate.point[0] - point[0], candidate.point[1] - point[1]),
    }))
    .filter((candidate) => candidate.distance <= options.tolerance_m)
    .sort((left, right) => left.distance - right.distance)[0];
  if (nearest) {
    point = nearest.point;
    guides.push(
      { axis: "x", value: point[0], label: nearest.label },
      { axis: "y", value: point[1], label: nearest.label },
    );
    label = nearest.label;
  }
  return { point, guides: uniqueGuides(guides), label };
}

export function distanceMeters(from: Point2, to: Point2): number {
  return Math.hypot(to[0] - from[0], to[1] - from[1]);
}

export function setWallLength(
  wall: WallEntity,
  lengthM: number,
  anchor: "start" | "end" = "start",
): { from: Point2; to: Point2 } | null {
  return setSegmentLength(wall.from, wall.to, lengthM, anchor);
}

export function setSegmentLength(
  from: Point2,
  to: Point2,
  lengthM: number,
  anchor: "start" | "end" = "start",
): { from: Point2; to: Point2 } | null {
  if (!Number.isFinite(lengthM) || lengthM < 0.05) return null;
  const currentLength = distanceMeters(from, to);
  if (currentLength < 1e-8) return null;
  const direction: Point2 = [
    (to[0] - from[0]) / currentLength,
    (to[1] - from[1]) / currentLength,
  ];
  if (anchor === "end") {
    return {
      from: [
        roundMetric(to[0] - direction[0] * lengthM),
        roundMetric(to[1] - direction[1] * lengthM),
      ],
      to: [...to],
    };
  }
  return {
    from: [...from],
    to: [
      roundMetric(from[0] + direction[0] * lengthM),
      roundMetric(from[1] + direction[1] * lengthM),
    ],
  };
}

export interface PolygonEditResult {
  polygon: Point2[];
  valid: boolean;
  reason: "ok" | "too_few_vertices" | "area_too_small" | "self_intersection";
}

export function movePolygonVertex(
  polygon: Point2[],
  index: number,
  point: Point2,
  minimumAreaM2 = 0.05,
): PolygonEditResult {
  if (polygon.length < 3 || index < 0 || index >= polygon.length) {
    return { polygon, valid: false, reason: "too_few_vertices" };
  }
  const next = polygon.map((vertex, vertexIndex) =>
    vertexIndex === index ? point : vertex,
  );
  if (!isSimplePolygon(next)) {
    return { polygon, valid: false, reason: "self_intersection" };
  }
  if (polygonArea(next) < minimumAreaM2) {
    return { polygon, valid: false, reason: "area_too_small" };
  }
  return { polygon: next, valid: true, reason: "ok" };
}

export function polygonArea(polygon: Point2[]): number {
  if (polygon.length < 3) return 0;
  const twiceArea = polygon.reduce((sum, point, index) => {
    const next = polygon[(index + 1) % polygon.length];
    return sum + point[0] * next[1] - next[0] * point[1];
  }, 0);
  return Math.abs(twiceArea) / 2;
}

export function isSimplePolygon(polygon: Point2[]): boolean {
  if (polygon.length < 3) return false;
  for (let first = 0; first < polygon.length; first += 1) {
    const firstNext = (first + 1) % polygon.length;
    for (let second = first + 1; second < polygon.length; second += 1) {
      const secondNext = (second + 1) % polygon.length;
      if (
        first === second ||
        firstNext === second ||
        secondNext === first
      ) continue;
      if (segmentsIntersect(
        polygon[first],
        polygon[firstNext],
        polygon[second],
        polygon[secondNext],
      )) return false;
    }
  }
  return true;
}

function snapCandidates(graph: PlanGraph, levelId: string): Candidate[] {
  const candidates: Candidate[] = [];
  for (const wall of graph.walls.filter((item) => item.level_id === levelId)) {
    candidates.push(
      { point: wall.from, label: "Wall endpoint" },
      { point: wall.to, label: "Wall endpoint" },
      {
        point: [(wall.from[0] + wall.to[0]) / 2, (wall.from[1] + wall.to[1]) / 2],
        label: "Wall midpoint",
      },
    );
  }
  for (const opening of graph.openings.filter((item) => item.level_id === levelId)) {
    candidates.push({ point: opening.center_m, label: "Opening center" });
  }
  for (const fixture of graph.fixtures.filter((item) => item.level_id === levelId)) {
    candidates.push({ point: fixture.center_m, label: "Object center" });
  }
  return candidates;
}

function uniqueGuides(guides: SmartGuide[]): SmartGuide[] {
  const seen = new Set<string>();
  return guides.filter((guide) => {
    const key = `${guide.axis}:${guide.value.toFixed(6)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function roundMetric(value: number): number {
  return Math.round(value * 1_000_000_000) / 1_000_000_000;
}

function segmentsIntersect(a: Point2, b: Point2, c: Point2, d: Point2): boolean {
  const orientation = (p: Point2, q: Point2, r: Point2) =>
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const first = orientation(a, b, c);
  const second = orientation(a, b, d);
  const third = orientation(c, d, a);
  const fourth = orientation(c, d, b);
  const epsilon = 1e-9;
  if (
    Math.abs(first) < epsilon ||
    Math.abs(second) < epsilon ||
    Math.abs(third) < epsilon ||
    Math.abs(fourth) < epsilon
  ) {
    return false;
  }
  return (first > 0) !== (second > 0) && (third > 0) !== (fourth > 0);
}
