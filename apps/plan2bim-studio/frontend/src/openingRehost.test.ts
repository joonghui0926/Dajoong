import { describe, expect, it } from "vitest";

import { planOpeningRehost } from "./openingRehost";
import type { PlanGraph } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1" }],
    walls: [
      { id: "wall-source", level_id: "L1", from: [0, 0], to: [8, 0], thickness_m: 0.12, height_m: 3 },
      { id: "wall-target", level_id: "L1", from: [8, 4], to: [0, 4], thickness_m: 0.12, height_m: 3 },
    ],
    rooms: [],
    openings: [
      {
        id: "door-1",
        level_id: "L1",
        type: "door",
        wall_id: "wall-source",
        center_m: [2, 0],
        x_m: 2,
        width_m: 0.9,
        height_m: 2.1,
        handing: "start",
        swing_side: "positive",
      },
      {
        id: "window-1",
        level_id: "L1",
        type: "window",
        wall_id: "wall-target",
        center_m: [4, 4],
        x_m: 4,
        width_m: 2,
        height_m: 1.2,
      },
    ],
    fixtures: [],
    routes: [],
    vertical_connections: [],
  };
}

describe("opening rehosting", () => {
  it("preserves world door orientation when the target wall direction is reversed", () => {
    const plan = planOpeningRehost(graph(), "door-1", "wall-target", [2, 4]);
    expect(plan.valid).toBe(true);
    expect(plan.changes).toMatchObject({
      wall_id: "wall-target",
      center_m: [2, 4],
      handing: "end",
      swing_side: "negative",
    });
  });

  it("moves to the closest clear span instead of failing at an occupied click", () => {
    const plan = planOpeningRehost(graph(), "door-1", "wall-target", [4, 4]);
    expect(plan.valid).toBe(true);
    expect((plan.changes.center_m as [number, number])[0]).not.toBe(4);
    expect(plan.notices).toContain("Opening shifted to the nearest clear wall span.");
  });

  it("rejects a target wall with no span large enough for the opening", () => {
    const model = graph();
    model.openings[1].width_m = 6.9;
    model.openings[1].center_m = [4, 4];
    const plan = planOpeningRehost(model, "door-1", "wall-target", [1, 4]);
    expect(plan).toMatchObject({ valid: false, reason: expect.stringContaining("no clear span") });
  });
});
