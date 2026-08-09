import { describe, expect, it } from "vitest";

import { planArrangement } from "./arrangementPlanner";
import type { PlanGraph, Selection } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1" }, { id: "L2", name: "Level 2" }],
    rooms: [{
      id: "room",
      level_id: "L1",
      name: "Open office",
      polygon: [[-2, -2], [14, -2], [14, 12], [-2, 12]],
    }],
    walls: [
      { id: "wall-left", level_id: "L1", from: [0, 0], to: [0, 10], thickness_m: 0.2, height_m: 3 },
      { id: "wall-right", level_id: "L1", from: [12, 0], to: [12, 10], thickness_m: 0.2, height_m: 3 },
    ],
    openings: [
      { id: "door-left", level_id: "L1", type: "door", wall_id: "wall-left", center_m: [0, 2], width_m: 0.9, height_m: 2.1 },
      { id: "door-right", level_id: "L1", type: "door", wall_id: "wall-right", center_m: [12, 7], width_m: 0.9, height_m: 2.1 },
    ],
    fixtures: [
      { id: "desk-a", level_id: "L1", type: "desk", center_m: [2, 2], size_m: [1, 1, 0.75] },
      { id: "desk-b", level_id: "L1", type: "desk", center_m: [5, 4], size_m: [2, 1, 0.75] },
      { id: "desk-key", level_id: "L1", type: "desk", center_m: [9, 7], size_m: [1, 2, 0.75] },
    ],
    routes: [],
    vertical_connections: [],
  };
}

const fixtureSelection: Selection[] = [
  { collection: "fixtures", id: "desk-a" },
  { collection: "fixtures", id: "desk-b" },
  { collection: "fixtures", id: "desk-key" },
];

describe("BIM-safe arrangement planner", () => {
  it("uses the last-selected component as the fixed key object", () => {
    const plan = planArrangement(graph(), fixtureSelection, { type: "align", mode: "center-y" });
    expect(plan.valid).toBe(true);
    expect(plan.keySelection).toEqual({ collection: "fixtures", id: "desk-key" });
    expect(plan.changesById).toMatchObject({
      "desk-a": { center_m: [2, 7], room_id: "room" },
      "desk-b": { center_m: [5, 7], room_id: "room" },
    });
    expect(plan.changesById["desk-key"]).toBeUndefined();
  });

  it("distributes equal clear gaps while keeping the outer anchors fixed", () => {
    const plan = planArrangement(graph(), fixtureSelection, { type: "distribute", axis: "horizontal" });
    expect(plan.valid).toBe(true);
    expect(plan.changesById["desk-b"]).toMatchObject({ center_m: [5.5, 4] });
    expect(plan.changesById["desk-a"]).toBeUndefined();
    expect(plan.changesById["desk-key"]).toBeUndefined();
  });

  it("projects openings onto their hosts and rejects a cross-level arrange", () => {
    const openingPlan = planArrangement(graph(), [
      { collection: "openings", id: "door-left" },
      { collection: "openings", id: "door-right" },
    ], { type: "align", mode: "center-y" });
    expect(openingPlan.valid).toBe(true);
    expect(openingPlan.changesById["door-left"]).toMatchObject({ center_m: [0, 7] });

    const crossLevel = graph();
    crossLevel.fixtures[1].level_id = "L2";
    expect(planArrangement(crossLevel, fixtureSelection.slice(0, 2), {
      type: "align",
      mode: "center-x",
    })).toMatchObject({ valid: false, reason: expect.stringContaining("one level") });
  });
});
