import {
  alignmentChanges,
  distributionChanges,
  type AlignmentMode,
  type DistributionAxis,
  type EntityChanges,
} from "./editorCommands";
import { validateFixtureTransformChanges } from "./fixturePlacement";
import { moveOpeningToPoint, validateOpeningPlacement } from "./openingGeometry";
import type {
  BaseEntity,
  CollectionName,
  OpeningEntity,
  PlanGraph,
  Selection,
} from "./types";

export type ArrangementOperation =
  | { type: "align"; mode: AlignmentMode }
  | { type: "distribute"; axis: DistributionAxis };

export interface ArrangementPlan {
  valid: boolean;
  selections: Selection[];
  changesById: EntityChanges;
  notices: string[];
  keySelection?: Selection;
  reason?: string;
}

const ARRANGEABLE_COLLECTIONS = new Set<CollectionName>([
  "openings",
  "fixtures",
  "vertical_connections",
]);

export function canArrangeSelection(selection: Selection): boolean {
  return ARRANGEABLE_COLLECTIONS.has(selection.collection);
}

export function planArrangement(
  graph: PlanGraph,
  selections: Selection[],
  operation: ArrangementOperation,
): ArrangementPlan {
  const minimum = operation.type === "align" ? 2 : 3;
  if (selections.length < minimum) {
    return invalid(operation.type === "align"
      ? "Select at least two components to align."
      : "Select at least three components to distribute.");
  }
  if (selections.some((selection) => !canArrangeSelection(selection))) {
    return invalid("Arrange supports doors, windows, placed objects, and vertical connections.");
  }

  const resolved = selections.map((selection) => ({
    selection,
    entity: findEntity(graph, selection),
  }));
  const missing = resolved.find((entry) => !entry.entity);
  if (missing) return invalid(`${missing.selection.id} is no longer present in the model.`);
  const entities = resolved.map((entry) => arrangementEntity(graph, entry.selection, entry.entity!));
  if (entities.some((entity) => !validPositionedEntity(entity))) {
    return invalid("Every selected component needs finite plan coordinates and a valid footprint.");
  }
  const workPlanes = new Set(entities.map(workPlaneId));
  if (workPlanes.size !== 1) {
    return invalid("Arrange selections must share one level or work plane.");
  }

  const requestedChanges = operation.type === "align"
    ? alignmentChanges(entities, operation.mode)
    : distributionChanges(entities, operation.axis);
  if (!Object.keys(requestedChanges).length) {
    return invalid(operation.type === "align"
      ? "The selected components are already aligned to the key object."
      : "The selected components already have equal clear spacing.");
  }

  const selectedOpeningIds = new Set(
    selections.filter((selection) => selection.collection === "openings").map((selection) => selection.id),
  );
  const fixedOpenings = graph.openings.filter((opening) => !selectedOpeningIds.has(opening.id));
  const openingChanges: EntityChanges = {};
  const notices: string[] = [];
  for (const entry of resolved) {
    if (entry.selection.collection !== "openings") continue;
    const requested = requestedChanges[entry.selection.id];
    if (!requested) continue;
    const opening = entry.entity as OpeningEntity;
    const target = requested.center_m as [number, number] | undefined;
    const host = graph.walls.find((wall) => wall.id === opening.wall_id);
    if (!target || !host) return invalid(`${opening.id} has no valid host wall.`);
    const placement = moveOpeningToPoint(opening, host, fixedOpenings, target);
    if (!placement.valid || !placement.changes) {
      return invalid(openingFailure(opening, placement.reason, placement.conflictId));
    }
    openingChanges[opening.id] = placement.changes;
    if (distance(placement.changes.center_m, target) > 0.001) {
      notices.push(`${opening.id} was projected onto its host wall.`);
    }
  }

  if (Object.keys(openingChanges).length) {
    const finalOpenings = graph.openings.map((opening) => ({
      ...opening,
      ...(openingChanges[opening.id] ?? {}),
    }));
    for (const openingId of Object.keys(openingChanges)) {
      const opening = finalOpenings.find((item) => item.id === openingId);
      const host = opening && graph.walls.find((wall) => wall.id === opening.wall_id);
      if (!opening || !host) return invalid(`${openingId} has no valid host wall.`);
      const validation = validateOpeningPlacement(opening, host, finalOpenings);
      if (!validation.valid) {
        return invalid(openingFailure(opening, validation.reason, validation.conflictId));
      }
    }
  }

  const changesById: EntityChanges = {
    ...requestedChanges,
    ...openingChanges,
  };
  const fixtureValidation = validateFixtureTransformChanges(graph, changesById);
  if (!fixtureValidation.valid) {
    return invalid(fixtureValidation.reason ?? "Arrange violates component clearance constraints.");
  }
  notices.push(...fixtureValidation.notices);

  const movedSelections = selections.filter((selection) => fixtureValidation.changesById[selection.id]);
  return {
    valid: true,
    selections: movedSelections,
    changesById: fixtureValidation.changesById,
    notices: unique(notices),
    keySelection: selections.at(-1),
  };
}

function findEntity(graph: PlanGraph, selection: Selection): BaseEntity | null {
  const collection = graph[selection.collection];
  if (!Array.isArray(collection)) return null;
  return (collection as BaseEntity[]).find((entity) => entity.id === selection.id) ?? null;
}

function arrangementEntity(
  graph: PlanGraph,
  selection: Selection,
  entity: BaseEntity,
): BaseEntity {
  if (selection.collection !== "openings") return entity;
  const opening = entity as OpeningEntity;
  const wall = graph.walls.find((item) => item.id === opening.wall_id);
  if (!wall) return entity;
  const yawDeg = Math.atan2(wall.to[1] - wall.from[1], wall.to[0] - wall.from[0]) * 180 / Math.PI;
  return {
    ...entity,
    size_m: [opening.width_m, Math.max(0.02, wall.thickness_m), opening.height_m],
    yaw_deg: yawDeg,
  };
}

function validPositionedEntity(entity: BaseEntity): boolean {
  const center = entity.center_m;
  const footprint = Array.isArray(entity.size_m) ? entity.size_m : entity.footprint_m;
  return Array.isArray(center)
    && center.length >= 2
    && center.slice(0, 2).every((value) => Number.isFinite(Number(value)))
    && Array.isArray(footprint)
    && footprint.length >= 2
    && footprint.slice(0, 2).every((value) => Number(value) > 0);
}

function workPlaneId(entity: BaseEntity): string {
  return String(entity.level_id ?? entity.from_level_id ?? "");
}

function openingFailure(
  opening: OpeningEntity,
  reason?: string,
  conflictId?: string,
): string {
  if (reason === "overlap") return `${opening.id} would overlap ${conflictId ?? "another opening"}.`;
  if (reason === "outside_wall") return `${opening.id} would leave its host wall.`;
  return `${opening.id} cannot move to the requested arrangement on its host wall.`;
}

function distance(left: [number, number], right: [number, number]): number {
  return Math.hypot(left[0] - right[0], left[1] - right[1]);
}

function unique(values: string[]): string[] {
  return [...new Set(values)];
}

function invalid(reason: string): ArrangementPlan {
  return { valid: false, selections: [], changesById: {}, notices: [], reason };
}
