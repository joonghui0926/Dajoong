import { collections, entities } from "./graph";
import { selectionKey } from "./editorViewState";
import type {
  BaseEntity,
  CollectionName,
  PlanGraph,
  PlanGraphViolation,
  QualificationClaim,
  Selection,
} from "./types";

export interface ReviewRiskReason {
  code: string;
  label: string;
  contribution: number;
}

export interface ReviewPriority {
  selection: Selection;
  score: number;
  percent: number;
  band: "low" | "medium" | "high";
  modelConfidence: number;
  reasons: ReviewRiskReason[];
}

const criticality: Partial<Record<CollectionName, [number, string]>> = {
  walls: [0.12, "Wall topology affects rooms and hosted elements"],
  openings: [0.14, "Opening geometry affects access and its host wall"],
  rooms: [0.10, "Room boundaries affect areas and contained objects"],
  fixtures: [0.06, "Installed object pose affects coordination"],
  routes: [0.14, "Building system routing affects coordination"],
  vertical_connections: [0.18, "Vertical circulation affects multiple levels"],
  constraints: [0.12, "A geometric constraint can move connected walls"],
  dimensions: [0.04, "A measured dimension supports downstream edits"],
};

const qualificationClaims: Partial<Record<CollectionName, string[]>> = {
  walls: ["wall_geometry"],
  rooms: ["room_polygon"],
  openings: ["opening_detection", "opening_host"],
  fixtures: ["fixed_object_pose"],
  routes: ["building_system_route"],
  vertical_connections: ["vertical_connection"],
};

const difficultyContribution: Record<string, number> = {
  simple: 0,
  moderate: 0.04,
  difficult: 0.08,
  extreme: 0.12,
};

const severityContribution: Record<string, number> = {
  info: 0.12,
  warning: 0.28,
  error: 0.55,
};

export function planReviewQueue(graph: PlanGraph): ReviewPriority[] {
  const violationsByEntity = new Map<string, PlanGraphViolation[]>();
  for (const violation of graph.verification?.violations ?? []) {
    for (const entityId of violation.entity_ids ?? []) {
      const list = violationsByEntity.get(entityId) ?? [];
      list.push(violation);
      violationsByEntity.set(entityId, list);
    }
  }
  const claims = new Map(
    (graph.qualification?.claims ?? []).map((claim) => [claim.claim, claim]),
  );
  const priorities: ReviewPriority[] = [];

  for (const collection of collections) {
    if (collection === "levels") continue;
    for (const entity of entities(graph, collection)) {
      const confidence = entityConfidence(entity);
      const violations = violationsByEntity.get(entity.id) ?? [];
      const relationshipReasons = relationshipRisk(graph, collection, entity);
      const requiresReview = entity.review_state !== "accepted"
        || confidence < 0.9
        || violations.length > 0
        || relationshipReasons.length > 0;
      if (!requiresReview) continue;

      const reasons: ReviewRiskReason[] = [];
      addReason(
        reasons,
        "model_uncertainty",
        `${Math.round(confidence * 100)}% entity confidence`,
        (1 - confidence) * 0.55,
      );
      if (entity.review_state === "rejected") {
        addReason(reasons, "review_state", "Entity is currently rejected", 0.25);
      } else if (entity.review_state !== "accepted") {
        addReason(reasons, "review_state", "Entity still requires human review", 0.16);
      }

      const difficulty = graph.drawing_profile?.difficulty_class
        ?? graph.qualification?.difficulty_class
        ?? "";
      addReason(
        reasons,
        "drawing_complexity",
        `${difficulty || "Unprofiled"} drawing complexity`,
        difficultyContribution[difficulty] ?? 0.06,
      );

      const categoryCriticality = criticality[collection];
      if (categoryCriticality) {
        addReason(reasons, "bim_criticality", categoryCriticality[1], categoryCriticality[0]);
      }

      const strongestViolation = [...violations].sort(
        (left, right) => (severityContribution[right.severity] ?? 0) - (severityContribution[left.severity] ?? 0),
      )[0];
      if (strongestViolation) {
        addReason(
          reasons,
          `violation:${strongestViolation.code}`,
          `${strongestViolation.severity.toUpperCase()}: ${strongestViolation.message}`,
          severityContribution[strongestViolation.severity] ?? 0.2,
        );
      }
      reasons.push(...relationshipReasons);

      const claimReason = qualificationRisk(collection, claims);
      if (claimReason) reasons.push(claimReason);
      if (graph.qualification?.exact_model_match === false) {
        addReason(reasons, "model_pair", "The active model pair does not match sealed evidence", 0.08);
      }

      reasons.sort((left, right) => right.contribution - left.contribution || left.label.localeCompare(right.label));
      const score = clamp01(reasons.reduce((sum, reason) => sum + reason.contribution, 0));
      priorities.push({
        selection: { collection, id: entity.id },
        score: round(score),
        percent: Math.round(score * 100),
        band: score >= 0.7 ? "high" : score >= 0.45 ? "medium" : "low",
        modelConfidence: round(confidence),
        reasons,
      });
    }
  }

  return priorities.sort(
    (left, right) => right.score - left.score
      || left.selection.collection.localeCompare(right.selection.collection)
      || left.selection.id.localeCompare(right.selection.id, undefined, { numeric: true }),
  );
}

export function reviewPriorityMap(priorities: ReviewPriority[]): Map<string, ReviewPriority> {
  return new Map(priorities.map((priority) => [selectionKey(priority.selection), priority]));
}

function relationshipRisk(
  graph: PlanGraph,
  collection: CollectionName,
  entity: BaseEntity,
): ReviewRiskReason[] {
  const reasons: ReviewRiskReason[] = [];
  if (collection === "openings") {
    const wallId = String(entity.wall_id ?? "");
    const wall = graph.walls.find((candidate) => candidate.id === wallId);
    if (!wall) {
      addReason(reasons, "missing_host_wall", "Opening has no valid host wall", 0.55);
    } else {
      const wallLength = Math.hypot(wall.to[0] - wall.from[0], wall.to[1] - wall.from[1]);
      if (Number(entity.width_m ?? 0) > wallLength + 1e-6) {
        addReason(reasons, "opening_outside_host", "Opening is wider than its host wall", 0.45);
      }
    }
  } else if (collection === "fixtures") {
    const roomId = String(entity.room_id ?? "");
    if (roomId && !graph.rooms.some((room) => room.id === roomId)) {
      addReason(reasons, "missing_room", "Object references a missing room", 0.4);
    } else if (!roomId && String(entity.mounting ?? "floor") === "floor") {
      addReason(reasons, "unassigned_room", "Floor object is not assigned to a room", 0.12);
    }
  } else if (collection === "vertical_connections") {
    const levelIds = new Set(graph.levels.map((level) => level.id));
    const from = String(entity.from_level_id ?? "");
    const to = String(entity.to_level_id ?? "");
    if (!levelIds.has(from) || !levelIds.has(to) || from === to) {
      addReason(reasons, "invalid_level_connection", "Vertical connection has invalid level endpoints", 0.6);
    }
  } else if (collection === "routes") {
    if (!Array.isArray(entity.points_m) || entity.points_m.length < 2) {
      addReason(reasons, "invalid_route", "Building system route has fewer than two points", 0.5);
    }
  } else if (collection === "rooms") {
    if (!Array.isArray(entity.polygon) || entity.polygon.length < 3) {
      addReason(reasons, "invalid_room", "Room boundary has fewer than three vertices", 0.5);
    }
  } else if (collection === "walls") {
    const from = point(entity.from);
    const to = point(entity.to);
    if (!from || !to || Math.hypot(to[0] - from[0], to[1] - from[1]) < 0.05) {
      addReason(reasons, "invalid_wall", "Wall has invalid or near-zero geometry", 0.5);
    }
  } else if (collection === "constraints") {
    const wallIds = new Set(graph.walls.map((wall) => wall.id));
    const references = Array.isArray(entity.references) ? entity.references : [];
    if (!references.length || references.some((reference) => {
      const record = reference as Record<string, unknown>;
      return !wallIds.has(String(record.entity_id ?? ""));
    })) {
      addReason(reasons, "invalid_constraint", "Constraint contains a missing wall reference", 0.5);
    }
  }
  return reasons;
}

function qualificationRisk(
  collection: CollectionName,
  claims: Map<string, QualificationClaim>,
): ReviewRiskReason | null {
  const relevant = (qualificationClaims[collection] ?? [])
    .map((name) => claims.get(name))
    .filter((claim): claim is QualificationClaim => Boolean(claim) && claim?.status !== "measured")
    .sort((left, right) => qualificationWeight(right) - qualificationWeight(left));
  const claim = relevant[0];
  if (!claim) return null;
  return {
    code: `qualification:${claim.claim}`,
    label: `${claim.claim.replaceAll("_", " ")} is ${claim.status.replaceAll("_", " ")}`,
    contribution: qualificationWeight(claim),
  };
}

function qualificationWeight(claim: QualificationClaim): number {
  if (claim.status === "unmeasured") return 0.10;
  if (claim.status === "model_mismatch") return 0.09;
  if (claim.status === "insufficient_sample") return 0.07;
  return 0;
}

function entityConfidence(entity: BaseEntity): number {
  if (typeof entity.confidence === "number" && Number.isFinite(entity.confidence)) {
    return clamp01(entity.confidence);
  }
  if (typeof entity.uncertainty === "number" && Number.isFinite(entity.uncertainty)) {
    return clamp01(1 - entity.uncertainty);
  }
  return 0.5;
}

function addReason(
  target: ReviewRiskReason[],
  code: string,
  label: string,
  contribution: number,
): void {
  if (contribution <= 0) return;
  target.push({ code, label, contribution: round(contribution) });
}

function point(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const x = Number(value[0]);
  const y = Number(value[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function round(value: number): number {
  return Math.round(value * 1000) / 1000;
}
