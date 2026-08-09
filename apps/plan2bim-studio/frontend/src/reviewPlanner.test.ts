import { describe, expect, it } from "vitest";

import { planReviewQueue, reviewPriorityMap } from "./reviewPlanner";
import { selectionKey } from "./editorViewState";
import type { PlanGraph } from "./types";

function baseGraph(): PlanGraph {
  return {
    schema_version: "test",
    levels: [{ id: "L1", name: "Level 1", elevation_m: 0 }],
    walls: [{ id: "wall-1", level_id: "L1", from: [0, 0], to: [4, 0], thickness_m: 0.12, height_m: 3, confidence: 0.95, review_state: "review_required" }],
    rooms: [{ id: "room-1", level_id: "L1", name: "Office", polygon: [[0, 0], [4, 0], [4, 4], [0, 4]], confidence: 0.95, review_state: "accepted" }],
    openings: [{ id: "door-1", level_id: "L1", type: "door", wall_id: "wall-1", center_m: [2, 0], width_m: 0.9, height_m: 2.1, confidence: 0.95, review_state: "review_required" }],
    fixtures: [{ id: "chair-1", level_id: "L1", type: "chair", room_id: "room-1", center_m: [2, 2], size_m: [0.5, 0.5, 0.8], confidence: 0.4, review_state: "review_required" }],
    routes: [],
    vertical_connections: [],
  };
}

describe("evidence-bound BIM review planning", () => {
  it("prioritizes deterministic errors above raw confidence alone", () => {
    const graph = baseGraph();
    graph.verification = {
      schema_version: "test",
      release_allowed: false,
      review_required: true,
      checked_invariants: 1,
      passed_invariants: 0,
      violations: [{
        code: "OVERLAPPING_OPENINGS",
        severity: "error",
        message: "Openings overlap.",
        entity_ids: ["door-1"],
      }],
    };
    const plan = planReviewQueue(graph);
    expect(plan[0].selection.id).toBe("door-1");
    expect(plan[0].band).toBe("high");
    expect(plan[0].reasons[0].code).toBe("violation:OVERLAPPING_OPENINGS");
  });

  it("raises review priority for difficult drawings without calling it accuracy", () => {
    const simple = baseGraph();
    simple.drawing_profile = { schema_version: "test", difficulty_class: "simple", complexity_score: 0.2, reasons: [] };
    const difficult = structuredClone(simple);
    difficult.drawing_profile = { schema_version: "test", difficulty_class: "difficult", complexity_score: 0.8, reasons: [] };
    const simpleWall = planReviewQueue(simple).find((item) => item.selection.id === "wall-1");
    const difficultWall = planReviewQueue(difficult).find((item) => item.selection.id === "wall-1");
    expect(difficultWall?.score).toBeGreaterThan(simpleWall?.score ?? 1);
    expect(difficultWall?.reasons.some((reason) => reason.code === "drawing_complexity")).toBe(true);
  });

  it("keeps accepted human-corrected entities out unless a deterministic error remains", () => {
    const graph = baseGraph();
    graph.rooms[0] = { ...graph.rooms[0], confidence: 1, uncertainty: 0, model_version: "human-correction", review_state: "accepted" };
    expect(planReviewQueue(graph).some((item) => item.selection.id === "room-1")).toBe(false);
    graph.verification = {
      schema_version: "test",
      release_allowed: false,
      review_required: true,
      checked_invariants: 1,
      passed_invariants: 0,
      violations: [{ code: "ROOM_SELF_INTERSECTION", severity: "error", message: "Room intersects itself.", entity_ids: ["room-1"] }],
    };
    expect(planReviewQueue(graph).some((item) => item.selection.id === "room-1")).toBe(true);
  });

  it("surfaces missing BIM relationships and builds a stable selection lookup", () => {
    const graph = baseGraph();
    graph.openings[0].wall_id = "missing-wall";
    const plan = planReviewQueue(graph);
    const priority = reviewPriorityMap(plan).get(selectionKey({ collection: "openings", id: "door-1" }));
    expect(priority?.reasons.some((reason) => reason.code === "missing_host_wall")).toBe(true);
    expect(priority?.percent).toBeGreaterThanOrEqual(70);
  });
});
