import type { BaseEntity, WallEntity } from "./types";

export type AlignmentMode =
  | "left"
  | "center-x"
  | "right"
  | "top"
  | "center-y"
  | "bottom";

export type DistributionAxis = "horizontal" | "vertical";
export type EntityChanges = Record<string, Record<string, unknown>>;

interface PositionedEntity {
  id: string;
  center: [number, number];
  footprint: [number, number];
}

export function alignmentChanges(
  entities: BaseEntity[],
  mode: AlignmentMode,
): EntityChanges {
  const positioned = entities.map(positionedEntity).filter((item) => item !== null);
  if (positioned.length < 2) return {};
  const key = positioned.at(-1)!;
  const left = key.center[0] - key.footprint[0] / 2;
  const right = key.center[0] + key.footprint[0] / 2;
  const top = key.center[1] - key.footprint[1] / 2;
  const bottom = key.center[1] + key.footprint[1] / 2;
  return Object.fromEntries(positioned.slice(0, -1).flatMap((item) => {
    const [x, y] = item.center;
    let center: [number, number];
    if (mode === "left") center = [roundMetric(left + item.footprint[0] / 2), y];
    else if (mode === "right") center = [roundMetric(right - item.footprint[0] / 2), y];
    else if (mode === "top") center = [x, roundMetric(top + item.footprint[1] / 2)];
    else if (mode === "bottom") center = [x, roundMetric(bottom - item.footprint[1] / 2)];
    else if (mode === "center-x") center = [roundMetric(key.center[0]), y];
    else center = [x, roundMetric(key.center[1])];
    return samePoint(center, item.center) ? [] : [[item.id, { center_m: center }]];
  }));
}

export function distributionChanges(
  entities: BaseEntity[],
  axis: DistributionAxis,
): EntityChanges {
  const positioned = entities.map(positionedEntity).filter((item) => item !== null);
  if (positioned.length < 3) return {};
  const coordinate = axis === "horizontal" ? 0 : 1;
  const ordered = [...positioned].sort(
    (left, right) => left.center[coordinate] - right.center[coordinate] || left.id.localeCompare(right.id),
  );
  const first = ordered[0];
  const last = ordered.at(-1)!;
  const leading = first.center[coordinate] - first.footprint[coordinate] / 2;
  const trailing = last.center[coordinate] + last.footprint[coordinate] / 2;
  const occupied = ordered.reduce((sum, item) => sum + item.footprint[coordinate], 0);
  const gap = (trailing - leading - occupied) / (ordered.length - 1);
  let cursor = leading + first.footprint[coordinate];
  return Object.fromEntries(ordered.slice(1, -1).flatMap((item) => {
    cursor += gap;
    const center: [number, number] = [...item.center];
    center[coordinate] = roundMetric(cursor + item.footprint[coordinate] / 2);
    cursor += item.footprint[coordinate];
    return samePoint(center, item.center) ? [] : [[item.id, { center_m: center }]];
  }));
}

export function offsetChanges(
  entities: BaseEntity[],
  delta: [number, number],
): EntityChanges {
  return Object.fromEntries(
    entities
      .map(positionedEntity)
      .filter((item) => item !== null)
      .map((item) => [
        item.id,
        { center_m: [item.center[0] + delta[0], item.center[1] + delta[1]] },
      ]),
  );
}

export function joinWallEndpointChanges(walls: WallEntity[]): EntityChanges {
  if (walls.length !== 2) return {};
  const endpoints = ["from", "to"] as const;
  let nearest: {
    leftKey: "from" | "to";
    rightKey: "from" | "to";
    distance: number;
  } | null = null;
  for (const leftKey of endpoints) {
    for (const rightKey of endpoints) {
      const distance = Math.hypot(
        walls[0][leftKey][0] - walls[1][rightKey][0],
        walls[0][leftKey][1] - walls[1][rightKey][1],
      );
      if (!nearest || distance < nearest.distance) {
        nearest = { leftKey, rightKey, distance };
      }
    }
  }
  if (!nearest) return {};
  const point: [number, number] = [
    roundMetric((walls[0][nearest.leftKey][0] + walls[1][nearest.rightKey][0]) / 2),
    roundMetric((walls[0][nearest.leftKey][1] + walls[1][nearest.rightKey][1]) / 2),
  ];
  return {
    [walls[0].id]: { [nearest.leftKey]: point },
    [walls[1].id]: { [nearest.rightKey]: point },
  };
}

export function cornerWallChanges(
  walls: WallEntity[],
  maximumEndpointMoveM = 25,
): EntityChanges {
  if (walls.length !== 2) return {};
  const intersection = lineIntersection(walls[0], walls[1]);
  if (!intersection) return {};
  const changes: EntityChanges = {};
  for (const wall of walls) {
    const fromDistance = Math.hypot(wall.from[0] - intersection[0], wall.from[1] - intersection[1]);
    const toDistance = Math.hypot(wall.to[0] - intersection[0], wall.to[1] - intersection[1]);
    const key = fromDistance <= toDistance ? "from" : "to";
    if (Math.min(fromDistance, toDistance) > maximumEndpointMoveM) return {};
    changes[wall.id] = { [key]: intersection };
  }
  return changes;
}

function lineIntersection(left: WallEntity, right: WallEntity): [number, number] | null {
  const x1 = left.from[0];
  const y1 = left.from[1];
  const x2 = left.to[0];
  const y2 = left.to[1];
  const x3 = right.from[0];
  const y3 = right.from[1];
  const x4 = right.to[0];
  const y4 = right.to[1];
  const denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denominator) < 1e-9) return null;
  const leftCross = x1 * y2 - y1 * x2;
  const rightCross = x3 * y4 - y3 * x4;
  return [
    roundMetric((leftCross * (x3 - x4) - (x1 - x2) * rightCross) / denominator),
    roundMetric((leftCross * (y3 - y4) - (y1 - y2) * rightCross) / denominator),
  ];
}

function roundMetric(value: number): number {
  const rounded = Math.round(value * 1_000_000_000) / 1_000_000_000;
  return Object.is(rounded, -0) ? 0 : rounded;
}

function samePoint(left: [number, number], right: [number, number]): boolean {
  return Math.abs(left[0] - right[0]) <= 1e-9 && Math.abs(left[1] - right[1]) <= 1e-9;
}

function positionedEntity(entity: BaseEntity): PositionedEntity | null {
  const center = entity.center_m;
  if (!Array.isArray(center) || center.length < 2) return null;
  const footprint = Array.isArray(entity.size_m)
    ? entity.size_m
    : Array.isArray(entity.footprint_m)
      ? entity.footprint_m
      : [0, 0];
  const width = Math.max(0, Number(footprint[0] ?? 0));
  const depth = Math.max(0, Number(footprint[1] ?? 0));
  const yaw = Number(entity.yaw_deg ?? 0) * Math.PI / 180;
  const cosine = Math.abs(Math.cos(yaw));
  const sine = Math.abs(Math.sin(yaw));
  return {
    id: entity.id,
    center: [Number(center[0]), Number(center[1])],
    footprint: [
      roundMetric(width * cosine + depth * sine),
      roundMetric(width * sine + depth * cosine),
    ],
  };
}
