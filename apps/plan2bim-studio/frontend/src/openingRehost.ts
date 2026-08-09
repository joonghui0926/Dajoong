import {
  findNearestOpeningPlacement,
  openingFrame,
  projectPointToWall,
  type Point2,
} from "./openingGeometry";
import type { PlanGraph, Selection } from "./types";

export interface OpeningRehostPlan {
  valid: boolean;
  selection: Selection;
  changes: Record<string, unknown>;
  notices: string[];
  reason?: string;
}

export function planOpeningRehost(
  graph: PlanGraph,
  openingId: string,
  targetWallId: string,
  desiredPoint: Point2,
): OpeningRehostPlan {
  const selection: Selection = { collection: "openings", id: openingId };
  const opening = graph.openings.find((item) => item.id === openingId);
  if (!opening) return failure(selection, `${openingId} is no longer present in the graph.`);
  const sourceWall = graph.walls.find((item) => item.id === opening.wall_id);
  const targetWall = graph.walls.find((item) => item.id === targetWallId);
  if (!sourceWall) return failure(selection, `${openingId} has no valid source wall.`);
  if (!targetWall) return failure(selection, `${targetWallId} is no longer present in the graph.`);
  if (targetWall.level_id !== opening.level_id) {
    return failure(selection, "Choose a host wall on the opening's current level.");
  }
  const placement = findNearestOpeningPlacement(
    opening,
    targetWall,
    graph.openings,
    desiredPoint,
  );
  if (!placement.valid || !placement.changes) {
    return failure(selection, placementFailure(placement.reason, placement.conflictId));
  }

  const changes: Record<string, unknown> = {
    wall_id: targetWall.id,
    level_id: targetWall.level_id,
    ...placement.changes,
  };
  if (opening.type === "door") {
    const orientation = preservedDoorOrientation(opening, sourceWall, targetWall);
    if (orientation) Object.assign(changes, orientation);
  }

  const notices: string[] = [];
  const projected = projectPointToWall(desiredPoint, targetWall);
  if (projected && distance(projected.point, placement.changes.center_m) > 0.02) {
    notices.push("Opening shifted to the nearest clear wall span.");
  }
  if (sourceWall.id !== targetWall.id) {
    notices.push(`Host changed from ${sourceWall.id} to ${targetWall.id}.`);
  }
  return { valid: true, selection, changes, notices };
}

function preservedDoorOrientation(
  opening: PlanGraph["openings"][number],
  sourceWall: PlanGraph["walls"][number],
  targetWall: PlanGraph["walls"][number],
): { handing: "start" | "end"; swing_side: "positive" | "negative" } | null {
  const sourceFrame = openingFrame(opening, sourceWall);
  const targetVector = wallDirections(targetWall);
  if (!sourceFrame || !targetVector) return null;
  const oldClosedDirection: Point2 = normalize([
    sourceFrame.latch[0] - sourceFrame.hinge[0],
    sourceFrame.latch[1] - sourceFrame.hinge[1],
  ]);
  const oldOpenDirection: Point2 = normalize([
    sourceFrame.openLeafEnd[0] - sourceFrame.hinge[0],
    sourceFrame.openLeafEnd[1] - sourceFrame.hinge[1],
  ]);
  return {
    handing: dot(targetVector.tangent, oldClosedDirection) >= 0 ? "start" : "end",
    swing_side: dot(targetVector.normal, oldOpenDirection) >= 0 ? "positive" : "negative",
  };
}

function wallDirections(wall: PlanGraph["walls"][number]) {
  const dx = wall.to[0] - wall.from[0];
  const dy = wall.to[1] - wall.from[1];
  const length = Math.hypot(dx, dy);
  if (length < 1e-8) return null;
  const tangent: Point2 = [dx / length, dy / length];
  return { tangent, normal: [-tangent[1], tangent[0]] as Point2 };
}

function normalize(vector: Point2): Point2 {
  const length = Math.max(1e-8, Math.hypot(vector[0], vector[1]));
  return [vector[0] / length, vector[1] / length];
}

function dot(first: Point2, second: Point2): number {
  return first[0] * second[0] + first[1] * second[1];
}

function distance(first: Point2, second: Point2): number {
  return Math.hypot(second[0] - first[0], second[1] - first[1]);
}

function placementFailure(reason?: string, conflictId?: string): string {
  if (reason === "overlap") {
    return `Target wall has no clear span${conflictId ? ` near ${conflictId}` : ""}.`;
  }
  if (reason === "outside_wall") return "Target wall is too short for this opening.";
  if (reason === "too_narrow") return "Opening width must remain at least 200 mm.";
  return "Target wall geometry is invalid.";
}

function failure(selection: Selection, reason: string): OpeningRehostPlan {
  return { valid: false, selection, changes: {}, notices: [], reason };
}
