import { describe, expect, it } from "vitest";

import { planWallTransform } from "./wallTransform";
import type { PlanGraph } from "./types";

function graph(): PlanGraph {
  return {
    schema_version: "buili.plan-graph.v2",
    levels: [{ id: "L1", name: "Level 1", nominal_height_m: 3 }],
    walls: [
      { id: "wall-bottom", level_id: "L1", from: [0, 0], to: [4, 0], thickness_m: 0.2, height_m: 3 },
      { id: "wall-right", level_id: "L1", from: [4, 0], to: [4, 4], thickness_m: 0.2, height_m: 3 },
      { id: "wall-top", level_id: "L1", from: [4, 4], to: [0, 4], thickness_m: 0.2, height_m: 3 },
      { id: "wall-left", level_id: "L1", from: [0, 4], to: [0, 0], thickness_m: 0.2, height_m: 3 },
    ],
    rooms: [{ id: "room-1", level_id: "L1", name: "Office", polygon: [[0, 0], [4, 0], [4, 4], [0, 4]] }],
    openings: [{
      id: "door-1",
      level_id: "L1",
      type: "door",
      wall_id: "wall-right",
      center_m: [4, 2],
      x_m: 2,
      width_m: 0.9,
      height_m: 2.1,
    }],
    fixtures: [{
      id: "panel-1",
      level_id: "L1",
      type: "electrical-panel",
      family_id: "dajoong:electrical:panelboard-recessed",
      mounting: "wall",
      host_wall_id: "wall-right",
      room_id: "room-1",
      center_m: [3.8, 3],
      size_m: [0.55, 0.18, 1.1],
      yaw_deg: 90,
      base_elevation_m: 0.4,
    }],
    routes: [],
    vertical_connections: [],
    constraints: [
      {
        id: "constraint-bottom-right",
        level_id: "L1",
        type: "coincident",
        references: [
          { collection: "walls", entity_id: "wall-bottom", handle: "to" },
          { collection: "walls", entity_id: "wall-right", handle: "from" },
        ],
      },
      {
        id: "constraint-right-top",
        level_id: "L1",
        type: "coincident",
        references: [
          { collection: "walls", entity_id: "wall-right", handle: "to" },
          { collection: "walls", entity_id: "wall-top", handle: "from" },
        ],
      },
    ],
  };
}

describe("wall transform planning", () => {
  it("moves a wall with its room boundary, opening, host fixture, and constrained endpoints", () => {
    const plan = planWallTransform(graph(), {
      "wall-right": { from: [5, 0], to: [5, 4] },
    });
    expect(plan.valid).toBe(true);
    expect(plan.graph?.rooms[0].polygon).toEqual([[0, 0], [5, 0], [5, 4], [0, 4]]);
    expect(plan.graph?.openings[0].center_m).toEqual([5, 2]);
    expect(plan.graph?.fixtures[0].center_m[0]).toBeCloseTo(4.8);
    expect(plan.graph?.walls.find((wall) => wall.id === "wall-bottom")?.to).toEqual([5, 0]);
    expect(plan.graph?.walls.find((wall) => wall.id === "wall-top")?.from).toEqual([5, 4]);
    expect(plan.entries.map((entry) => entry.selection.collection)).toEqual(
      expect.arrayContaining(["walls", "rooms", "openings", "fixtures"]),
    );
  });

  it("rejects a degenerate wall and a self intersecting room boundary", () => {
    expect(planWallTransform(graph(), {
      "wall-right": { from: [4, 0], to: [4, 0.01] },
    })).toMatchObject({ valid: false, reason: expect.stringContaining("50 mm") });

    expect(planWallTransform(graph(), {
      "wall-right": { from: [4, 5], to: [4, -1] },
    })).toMatchObject({ valid: false, reason: expect.stringContaining("self intersecting") });
  });

  it("rejects a shortened host wall when its openings overlap after synchronization", () => {
    const model = graph();
    model.openings.push({
      id: "door-2",
      level_id: "L1",
      type: "door",
      wall_id: "wall-right",
      center_m: [4, 3.2],
      x_m: 3.2,
      width_m: 0.9,
      height_m: 2.1,
    });
    const plan = planWallTransform(model, {
      "wall-right": { from: [4, 0], to: [4, 1.5] },
    });
    expect(plan).toMatchObject({ valid: false, reason: expect.stringMatching(/overlap|fit inside/) });
  });

  it("does not let an unrelated source boundary anomaly freeze normal wall editing", () => {
    const model = graph();
    model.rooms.push({
      id: "room-source-review",
      level_id: "L1",
      name: "Needs review",
      polygon: [[10, 10], [12, 12], [10, 12], [12, 10]],
    });
    const plan = planWallTransform(model, {
      "wall-right": { from: [4.25, 0], to: [4.25, 4] },
    });
    expect(plan.valid).toBe(true);
    expect(plan.graph?.rooms.find((room) => room.id === "room-source-review")?.polygon).toEqual(
      [[10, 10], [12, 12], [10, 12], [12, 10]],
    );
  });
});
